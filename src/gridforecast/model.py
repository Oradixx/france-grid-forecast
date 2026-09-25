"""Train, backtest, gate and serve the day-ahead consumption model.

Model registry = files on the `data` branch:
    lake/models/<version>.joblib    the fitted model
    lake/models/registry.json       every trained version, its metrics and whether it was promoted
    lake/models/current.json        the version the daily forecast uses
"""
import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import FEATURES, TARGET, build

log = logging.getLogger(__name__)

PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
              l2_regularization=1.0, random_state=0)

# Quality gate for a new model, on the last weeks it was not trained on
HOLDOUT_DAYS = 56
GATE_MAX_MAPE = 6.0             # % : above this, the model is not usable
GATE_MIN_GAIN_VS_NAIVE = 0.25   # must be at least 25 % better than "same hour last week"


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    return float(np.mean(np.abs(yhat[ok] - y[ok]) / y[ok]) * 100) if ok.any() else None


def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    return float(np.mean(np.abs(yhat[ok] - y[ok]))) if ok.any() else None


def scores(frame, pred_col):
    """Our model, RTE's day-ahead forecast and the seasonal naive baseline on the same hours."""
    f = frame.dropna(subset=[TARGET, pred_col, 'rte_forecast_d1_mw', 'lag_168h'])
    return {
        'hours': int(len(f)),
        'model': {'mape': mape(f[TARGET], f[pred_col]), 'mae_mw': mae(f[TARGET], f[pred_col])},
        'rte_d1': {'mape': mape(f[TARGET], f['rte_forecast_d1_mw']), 'mae_mw': mae(f[TARGET], f['rte_forecast_d1_mw'])},
        'naive_last_week': {'mape': mape(f[TARGET], f['lag_168h']), 'mae_mw': mae(f[TARGET], f['lag_168h'])},
    }


def training_rows(feats, end=None, start=None):
    rows = feats.dropna(subset=[TARGET] + FEATURES)
    rows = rows[~rows['is_interpolated'].fillna(False).astype(bool)]
    if start is not None:
        rows = rows[rows.index >= pd.Timestamp(start, tz='UTC')]
    if end is not None:
        rows = rows[rows.index < pd.Timestamp(end, tz='UTC')]
    return rows


def fit(rows):
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(rows[FEATURES], rows[TARGET])
    return model


def backtest(df, start=None):
    """Train on everything before `start`, predict every hour after it.

    Caveat, written in the report: the backtest uses *observed* temperatures, while the live
    forecast uses a weather forecast. It is therefore an optimistic bound; the live track record
    on the dashboard is the honest measure.
    """
    start = start or config.BACKTEST_START
    feats = build(df)
    model = fit(training_rows(feats, end=start))
    test = training_rows(feats, start=start).copy()
    test['yhat'] = model.predict(test[FEATURES])
    local = test.index.tz_convert(config.PARIS)
    monthly = []
    for month, part in test.groupby(local.strftime('%Y-%m')):
        s = scores(part, 'yhat')
        monthly.append({'month': month, **{k: s[k]['mape'] for k in ('model', 'rte_d1', 'naive_last_week')}})
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'train_until': start, 'test_from': start,
        'test_until': test.index.max().isoformat() if len(test) else None,
        'weather_note': 'observed temperatures (optimistic compared with live forecasts)',
        'overall': scores(test, 'yhat'),
        'monthly_mape': monthly,
    }
    return report, test


def train_candidate(df, now=None):
    """Fit on all data except the holdout, score the holdout, then refit on everything."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    feats = build(df)
    rows = training_rows(feats)
    last = rows.index.max()
    holdout_start = (last - pd.Timedelta(days=HOLDOUT_DAYS)).floor('D')
    model = fit(rows[rows.index < holdout_start])
    hold = rows[rows.index >= holdout_start].copy()
    hold['yhat'] = model.predict(hold[FEATURES])
    s = scores(hold, 'yhat')
    naive, ours = s['naive_last_week']['mape'], s['model']['mape']
    reasons = []
    if ours is None or ours > GATE_MAX_MAPE:
        reasons.append(f'holdout MAPE {ours} % above {GATE_MAX_MAPE} %')
    if ours is not None and naive is not None and ours > naive * (1 - GATE_MIN_GAIN_VS_NAIVE):
        reasons.append(f'not {GATE_MIN_GAIN_VS_NAIVE:.0%} better than the naive baseline ({ours:.2f} % vs {naive:.2f} %)')
    final = fit(rows)  # the promoted model also learns from the most recent weeks
    version = f"{last.strftime('%Y%m%d')}-{hashlib.sha1(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:6]}"
    meta = {
        'version': version, 'trained_at': now.isoformat(timespec='seconds'),
        'data_until': last.isoformat(), 'rows': int(len(rows)),
        'holdout': {'from': holdout_start.isoformat(), 'days': HOLDOUT_DAYS, **s},
        'gate': {'passed': not reasons, 'reasons': reasons,
                 'rules': {'max_mape': GATE_MAX_MAPE, 'min_gain_vs_naive': GATE_MIN_GAIN_VS_NAIVE}},
        'params': PARAMS, 'features': FEATURES, 'sklearn_version': sklearn.__version__,
    }
    return final, meta


def promote(model, meta, models_dir=None):
    """Save the version, record it in the registry, and point current.json at it if it passed."""
    d = models_dir or config.MODELS
    d.mkdir(parents=True, exist_ok=True)
    reg_path = d / 'registry.json'
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else []
    registry = [r for r in registry if r['version'] != meta['version']] + [meta]
    reg_path.write_text(json.dumps(registry, indent=2))
    if not meta['gate']['passed']:
        log.warning('model %s rejected by the gate: %s', meta['version'], '; '.join(meta['gate']['reasons']))
        return False
    joblib.dump(model, d / f"{meta['version']}.joblib")
    (d / 'current.json').write_text(json.dumps(meta, indent=2))
    # keep the three latest model files
    for old in sorted(d.glob('*.joblib'))[:-3]:
        old.unlink()
    log.info('model %s promoted', meta['version'])
    return True


def load_current(models_dir=None):
    d = models_dir or config.MODELS
    meta = json.loads((d / 'current.json').read_text())
    if meta['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(f"model saved with scikit-learn {meta['sklearn_version']}, running {sklearn.__version__}")
    return joblib.load(d / f"{meta['version']}.joblib"), meta
