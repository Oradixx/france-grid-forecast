"""Build the static dashboard: site/index.html + a data.json generated from the lake."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .features import TARGET


def _series(df, cols, since):
    part = df.loc[df.index >= since, cols]
    return {
        'hours': [t.tz_convert(config.PARIS).strftime('%Y-%m-%dT%H:%M') for t in part.index],
        **{c: [None if pd.isna(v) else round(float(v)) for v in part[c]] for c in cols},
    }


def _dbt_results(path):
    """Test outcomes of the last `dbt build` (run_results.json)."""
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    tests = [{'name': r['unique_id'].split('.')[2], 'status': r['status'],
              'failures': r.get('failures')} for r in res['results'] if r['unique_id'].startswith('test.')]
    return {'generated_at': res['metadata']['generated_at'], 'tests': tests,
            'passed': sum(t['status'] == 'pass' for t in tests), 'total': len(tests)}


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def build(df, out, dbt_target=None):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(config.ROOT / 'site', out)

    measured = df[TARGET].dropna()
    last = measured.index.max()
    since = (last - pd.Timedelta(days=14)).floor('D')
    chart = _series(df, [TARGET, 'rte_forecast_d1_mw'], since)

    forecasts = []
    if config.FORECASTS.exists():
        fc = pd.read_parquet(config.FORECASTS)
        fc['target_hour_utc'] = pd.to_datetime(fc['target_hour_utc'], utc=True)
        # the latest issue for each target hour
        fc = fc.sort_values('issue_date').drop_duplicates('target_hour_utc', keep='last')
        fc = fc[fc['target_hour_utc'] >= since]
        forecasts = [{'hour': t.tz_convert(config.PARIS).strftime('%Y-%m-%dT%H:%M'), 'mw': float(v),
                      'rte': None if pd.isna(r) else float(r), 'issue_date': d}
                     for t, v, r, d in zip(fc['target_hour_utc'], fc['forecast_mw'], fc['rte_forecast_d1_mw'], fc['issue_date'])]

    current = _read(config.MODELS / 'current.json')
    registry = _read(config.MODELS / 'registry.json') or []
    data = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'last_measured_hour': last.tz_convert(config.PARIS).strftime('%Y-%m-%dT%H:%M'),
        'chart': chart,
        'forecasts': forecasts,
        'live': _read(config.REPORTS / 'live.json'),
        'backtest': _read(config.REPORTS / 'backtest.json'),
        'model': current and {k: current[k] for k in ('version', 'trained_at', 'data_until', 'holdout', 'gate', 'rows')},
        'registry': [{'version': r['version'], 'trained_at': r['trained_at'], 'passed': r['gate']['passed'],
                      'holdout_mape': r['holdout']['model']['mape'], 'reasons': r['gate']['reasons']}
                     for r in registry[-10:]],
        'data_quality': _dbt_results(Path(dbt_target or config.ROOT / 'dbt' / 'target') / 'run_results.json'),
    }
    (out / 'data.json').write_text(json.dumps(data, default=str))
    return out
