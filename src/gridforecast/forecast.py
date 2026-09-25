"""Daily forecast and its evaluation.

forecasts.parquet is an append-only log: one row per (issue date, target hour), with our forecast,
RTE's day-ahead forecast and the naive baseline as known at issue time, and the temperature
forecast used. Past forecasts are never recomputed: the track record is what was really
published, not what today's model would say.
"""
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from . import config
from .features import FEATURES, TARGET, build, target_hours
from .model import load_current, mape

log = logging.getLogger(__name__)


def predict(df, issue_date=None, models_dir=None):
    issue_date = pd.Timestamp(issue_date or datetime.now(config.PARIS).date())
    model, meta = load_current(models_dir)
    feats = build(df)
    hours = target_hours(issue_date)
    rows = feats.reindex(hours)
    missing = rows[FEATURES].isna().any(axis=1)
    if missing.any():
        cols = rows[FEATURES].columns[rows[FEATURES].isna().any()].tolist()
        raise RuntimeError(f'{int(missing.sum())} target hours miss features {cols}: no forecast published')
    out = pd.DataFrame({
        'issue_date': issue_date.date().isoformat(),
        'issued_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'target_hour_utc': hours,
        'forecast_mw': model.predict(rows[FEATURES]).round(0),
        'rte_forecast_d1_mw': rows['rte_forecast_d1_mw'].to_numpy(),
        'naive_mw': rows['lag_168h'].to_numpy(),
        'temperature_c': rows['temperature_c'].round(2).to_numpy(),
        'model_version': meta['version'],
    })
    log.info('forecast for %s: %d hours, peak %.0f MW', hours[0].tz_convert(config.PARIS).date(),
             len(out), out['forecast_mw'].max())
    return out


def append(new, path=None, replace=False):
    """Add a day's forecast to the log. A published forecast is never rewritten: re-running the
    pipeline later the same day (with more data) would otherwise improve the track record after
    the fact. `replace=True` exists for repairing a broken run, and is logged."""
    path = path or config.FORECASTS
    path.parent.mkdir(parents=True, exist_ok=True)
    day = new['issue_date'].iloc[0]
    if path.exists():
        old = pd.read_parquet(path)
        if (old['issue_date'] == day).any():
            if not replace:
                log.info('forecast issued on %s already published, kept as it was', day)
                return old
            log.warning('replacing the forecast issued on %s', day)
            old = old[old['issue_date'] != day]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(['target_hour_utc', 'issue_date']).to_parquet(path, index=False)
    return new


def evaluate(df, path=None):
    """Score every published forecast whose hours have been measured since."""
    path = path or config.FORECASTS
    if not path.exists():
        return {'days': [], 'summary': {}}
    fc = pd.read_parquet(path)
    fc['target_hour_utc'] = pd.to_datetime(fc['target_hour_utc'], utc=True)
    actual = df[[TARGET, 'rte_forecast_d1_mw']].rename(columns={TARGET: 'actual_mw', 'rte_forecast_d1_mw': 'rte_now'})
    m = fc.merge(actual, left_on='target_hour_utc', right_index=True, how='left')
    # RTE may publish its day-ahead forecast after our run: score the version it published
    m['rte_forecast_d1_mw'] = m['rte_now'].fillna(m['rte_forecast_d1_mw'])
    m['day'] = m['target_hour_utc'].dt.tz_convert(config.PARIS).dt.date.astype(str)
    days = []
    for day, g in m.groupby('day'):
        done = g.dropna(subset=['actual_mw'])
        if len(done) < len(g):  # only fully measured days
            continue
        days.append({'day': day, 'model': mape(done['actual_mw'], done['forecast_mw']),
                     'rte_d1': mape(done['actual_mw'], done['rte_forecast_d1_mw']),
                     'naive_last_week': mape(done['actual_mw'], done['naive_mw']),
                     'model_version': g['model_version'].iloc[0]})
    scored = m.dropna(subset=['actual_mw'])
    summary = {}
    for window, label in [(7, 'last_7_days'), (30, 'last_30_days'), (None, 'since_launch')]:
        keep = [d['day'] for d in days][-window:] if window else [d['day'] for d in days]
        s = scored[scored['day'].isin(keep)]
        if len(s):
            summary[label] = {'days': len(keep), 'model': mape(s['actual_mw'], s['forecast_mw']),
                              'rte_d1': mape(s['actual_mw'], s['rte_forecast_d1_mw']),
                              'naive_last_week': mape(s['actual_mw'], s['naive_mw'])}
    return {'days': days, 'summary': summary}


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
