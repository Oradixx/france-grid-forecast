"""P10/P90 interval of the day-ahead forecast: quantile regression, calibrated on hours it did not see.

P10 and P90 are quantiles: consumption should fall below P10 one hour in ten and above P90 one hour
in ten, so the band between them holds 80 % of the hours. (Not the convention of wind yield studies,
where "P90" is the value exceeded 90 % of the time.)

Two gradient-boosting models give them. They share everything with the point model (the features
of features.build, the target relative to the 7-day level, the hyperparameters, the recency weights)
except the loss: the pinball loss of the 10th or the 90th percentile instead of the squared error.
The point model itself is not changed: the band is a separate estimate, so the point forecast can,
rarely, fall outside it.

Quantile boosting alone is over-confident: on real data, the raw band held 64 % of the hours of a
year it had not seen (2024) instead of 80 %, because the trees fit the hours they were trained on
more tightly than new ones. So each bound is calibrated on hours the models did not see
(conformalized quantile regression, Romano et al. 2019, one tail at a time): fit without the last
year, move each bound by what makes it miss 10 % of that year's hours, refit on everything and keep
the moves. Training and backtest run this one procedure, and prediction uses its result.

Registry, next to the point model's:
    lake/models/intervals/<version>.joblib   the two quantile models and their calibration
    lake/models/intervals/current.json       the version the daily forecast uses
"""
import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config, model
from .features import LEVEL, TARGET, build

log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.9)
NOMINAL_COVERAGE_PCT = round((QUANTILES[1] - QUANTILES[0]) * 100)
# Calibrate on the last year of the training data: every season counts once in the moves
CALIBRATION_DAYS = 365
# Misses come in runs (a cold spell, a holiday week), so the uncertainty of the coverage is
# estimated by resampling whole weeks of hours, not single hours
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_SAMPLES = 2000


def code_signature():
    """Short hash of what defines the interval models; includes the point model's signature, whose
    features and hyperparameters they share."""
    return hashlib.sha1(json.dumps([model.code_signature(), QUANTILES, CALIBRATION_DAYS, 'cqr']).encode()).hexdigest()[:6]


def _dir(models_dir=None):
    return (models_dir or config.MODELS) / 'intervals'


def is_stale(models_dir=None):
    """True when there are no interval models, or they were trained by different code."""
    cur = _dir(models_dir) / 'current.json'
    return not cur.exists() or not json.loads(cur.read_text())['version'].endswith(code_signature())


def pinball(y, q, tau):
    """Mean pinball loss of q as a forecast of the `tau` quantile of y, in the unit of y. It is
    lowest, in expectation, for the true quantile: it rewards a calibrated bound and a narrow one."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def _move(scores, alpha):
    """How far to move a bound so that at most `alpha` of the calibration hours lie beyond it: the
    empirical (1 - alpha) quantile of how far beyond it they are, with the (n + 1) correction of
    split conformal prediction (negligible with a year of hours)."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method='higher'))


class IntervalModel:
    """The P10 and P90 models, and how far the calibration moves each bound outwards (negative:
    inwards), as a share of the 7-day level so that the move scales with consumption: about 1 % is
    400 MW in summer and 700 MW in winter."""

    def __init__(self):
        self.bounds = [model.RelativeModel(**model.PARAMS, loss='quantile', quantile=q) for q in QUANTILES]
        self.moves = (0.0, 0.0)

    def fit(self, rows):
        # the point model's recency weights (model.fit): older years count less
        age_years = (rows.index.max() - rows.index).days / 365.25
        weights = np.power(0.5, age_years / model.RECENCY_HALF_LIFE_YEARS)
        for b in self.bounds:
            b.fit(rows, rows[TARGET], rows[LEVEL], sample_weight=weights)
        return self

    def unsorted(self, X):
        """P10 and P90 in MW, as the two models give them: separate models can cross."""
        level = X[LEVEL].to_numpy()
        return self.bounds[0].predict(X) - self.moves[0] * level, self.bounds[1].predict(X) + self.moves[1] * level

    def predict(self, X):
        lo, hi = self.unsorted(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def fit(rows):
    """Fit without the last CALIBRATION_DAYS of `rows`, move each bound so that it misses 10 % of
    those hours, refit on all rows and keep the moves. Returns the model and what the calibration saw."""
    cal_start = (rows.index.max() - pd.Timedelta(days=CALIBRATION_DAYS)).floor('D')
    if not (rows.index < cal_start).any():
        raise ValueError(f'interval models need more than {CALIBRATION_DAYS} days of training data')
    first = IntervalModel().fit(rows[rows.index < cal_start])
    cal = rows[rows.index >= cal_start]
    lo, hi = first.unsorted(cal)
    y, level = cal[TARGET].to_numpy(), cal[LEVEL].to_numpy()
    moves = (_move((lo - y) / level, QUANTILES[0]), _move((y - hi) / level, 1 - QUANTILES[1]))
    final = IntervalModel().fit(rows)
    final.moves = moves
    raw = scores(y, lo, hi)
    calibration = {'from': cal_start.isoformat(), 'until': rows.index.max().isoformat(), 'hours': int(len(cal)),
                   'uncalibrated': {k: raw[k] for k in ('coverage_pct', 'below_p10_pct', 'above_p90_pct')},
                   'p10_moved_down_pct': moves[0] * 100, 'p90_moved_up_pct': moves[1] * 100}
    return final, calibration


def _coverage_ci(inside, index):
    """95 % interval of the coverage by block bootstrap: resample whole weeks. Hourly misses are
    correlated over several days, so a binomial interval over hours would be several times too narrow."""
    block = np.asarray((index - index.min()) // pd.Timedelta(days=BOOTSTRAP_BLOCK_DAYS))
    hits, hours = np.bincount(block, weights=inside), np.bincount(block)
    hits, hours = hits[hours > 0], hours[hours > 0]
    pick = np.random.default_rng(0).integers(0, len(hours), (BOOTSTRAP_SAMPLES, len(hours)))
    coverage = hits[pick].sum(axis=1) / hours[pick].sum(axis=1) * 100
    return [float(v) for v in np.percentile(coverage, [2.5, 97.5])]


def scores(y, lo, hi, index=None):
    """How the band [lo, hi] did: share of hours inside it (bounds included) with its 95 % interval
    when `index` gives the hours, share below P10 and above P90 (each should be 10 %: an 80 % coverage
    can hide a band that sits too high), width, and the pinball loss of each bound."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    inside = (y >= lo) & (y <= hi)
    p10, p90 = pinball(y, lo, QUANTILES[0]), pinball(y, hi, QUANTILES[1])
    return {
        'hours': int(len(y)),
        'coverage_pct': float(inside.mean() * 100),
        'coverage_ci95_pct': None if index is None else _coverage_ci(inside, index),
        'below_p10_pct': float(np.mean(y < lo) * 100),
        'above_p90_pct': float(np.mean(y > hi) * 100),
        'mean_width_mw': float(np.mean(hi - lo)),
        'mean_width_pct': float(np.mean((hi - lo) / y) * 100),
        'pinball_p10_mw': p10, 'pinball_p90_mw': p90, 'pinball_mw': (p10 + p90) / 2,
    }


def backtest(df, test, start=None):
    """The interval part of the backtest: same training data and same test hours as model.backtest
    (`test` is its test frame, with the point forecast in `yhat`).

    Two baselines, scored the same way:
    - naive band: last week's value times the 10th and 90th percentiles of the week-on-week ratio in
      the training data (the interval counterpart of the naive point baseline);
    - constant band: our point forecast times the 10th and 90th percentiles of its error ratio over
      the calibration year (split conformal prediction around the point forecast). It is as wide, in
      relative terms, in a cold January as in a mild June; the quantile models are there to adapt.

    Same caveat as the point backtest: observed temperatures, so the band has not met the error of
    a weather forecast and the live coverage will be lower.
    """
    start = start or config.BACKTEST_START
    rows = model.training_rows(build(df), end=start)
    im, calibration = fit(rows)
    lo, hi = im.unsorted(test)
    crossed = int((lo > hi).sum())
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    y, yhat = test[TARGET].to_numpy(), test['yhat'].to_numpy()

    week_ratio = rows[TARGET] / rows['lag_168h']
    naive = [test['lag_168h'].to_numpy() * week_ratio.quantile(q) for q in QUANTILES]
    cal_start = pd.Timestamp(calibration['from'])
    cal = rows[rows.index >= cal_start]
    error_ratio = cal[TARGET] / model.fit(rows[rows.index < cal_start]).predict(cal)
    constant = [yhat * error_ratio.quantile(q) for q in QUANTILES]

    monthly = []
    local = test.index.tz_convert(config.PARIS).strftime('%Y-%m')
    for month in sorted(set(local)):
        m = np.asarray(local == month)
        s = scores(y[m], lo[m], hi[m])
        monthly.append({'month': month,
                        **{k: s[k] for k in ('hours', 'coverage_pct', 'below_p10_pct', 'above_p90_pct', 'mean_width_mw', 'pinball_mw')},
                        'constant_band_coverage_pct': scores(y[m], constant[0][m], constant[1][m])['coverage_pct'],
                        'naive_band_coverage_pct': scores(y[m], naive[0][m], naive[1][m])['coverage_pct']})
    return {
        'signature': code_signature(),
        'quantiles': list(QUANTILES), 'nominal_coverage_pct': NOMINAL_COVERAGE_PCT,
        'calibration': calibration,
        'overall': {'model': scores(y, lo, hi, test.index),
                    'constant_band': scores(y, *constant, test.index),
                    'naive_band': scores(y, *naive, test.index)},
        'crossed_hours': crossed,
        'point_outside_band_pct': float(np.mean((yhat < lo) | (yhat > hi)) * 100),
        'monthly': monthly,
    }


def train(df, now=None):
    """Interval models for the daily forecast, on all the data, like the promoted point model. There
    is no gate: the band never stops a forecast, and the backtest reports how well it holds."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    rows = model.training_rows(build(df))
    im, calibration = fit(rows)
    last = rows.index.max()
    meta = {
        'version': f"{last.strftime('%Y%m%d')}-{code_signature()}", 'trained_at': now.isoformat(timespec='seconds'),
        'data_until': last.isoformat(), 'rows': int(len(rows)), 'quantiles': list(QUANTILES),
        'calibration': calibration, 'params': model.PARAMS, 'point_model_signature': model.code_signature(),
        'sklearn_version': sklearn.__version__,
    }
    return im, meta


def save(im, meta, models_dir=None):
    d = _dir(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(im, d / f"{meta['version']}.joblib")
    (d / 'current.json').write_text(json.dumps(meta, indent=2))
    # keep the three latest model files
    for old in sorted(d.glob('*.joblib'))[:-3]:
        old.unlink()
    log.info('interval models %s saved', meta['version'])


def load_current(models_dir=None):
    d = _dir(models_dir)
    meta = json.loads((d / 'current.json').read_text())
    if not meta['version'].endswith(code_signature()):
        raise RuntimeError(f"interval models {meta['version']} were trained by other code: run `gridforecast train-intervals`")
    if meta['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(f"interval models saved with scikit-learn {meta['sklearn_version']}, running {sklearn.__version__}")
    return joblib.load(d / f"{meta['version']}.joblib"), meta
