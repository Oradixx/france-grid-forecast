import json

import duckdb
import pandas as pd
import pytest

from gridforecast import config, forecast, model, site


def test_dbt_marts(lake):
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    # the history wins over the real-time copy of the same timestamps (+1 % in the fixture)
    rank = con.sql("select nature_rank from fct_grid_hourly where hour_utc = '2026-06-24 10:00'").fetchone()[0]
    assert rank == 2
    # the autumn DST gap of the source is interpolated, and only that hour
    assert con.sql('select count(*) from fct_grid_hourly where is_interpolated').fetchone()[0] == 1
    # the spring DST phantom rows are dropped: one row per 15-minute timestamp
    n, distinct = con.sql('select count(*), count(distinct ts_utc) from stg_eco2mix').fetchone()
    assert n == distinct


def test_backtest_reports_all_three_methods(grid):
    report, test = model.backtest(grid, start='2026-03-01')
    o = report['overall']
    assert o['hours'] > 2000
    assert {'model', 'rte_d1', 'naive_last_week'} <= set(o)
    # a model that cannot beat "same hour last week" would be useless
    assert o['model']['mape'] < o['naive_last_week']['mape']


def test_gate_promotes_and_rejects(grid, lake, monkeypatch):
    m, meta = model.train_candidate(grid)
    assert meta['gate']['passed']
    assert model.promote(m, meta)
    assert (config.MODELS / 'current.json').exists()
    # an impossible gate: the candidate is recorded but not promoted
    monkeypatch.setattr(model, 'GATE_MAX_MAPE', 0.01)
    m2, meta2 = model.train_candidate(grid)
    meta2['version'] = 'rejected-version'
    assert not model.promote(m2, meta2)
    current = json.loads((config.MODELS / 'current.json').read_text())
    assert current['version'] == meta['version']
    registry = json.loads((config.MODELS / 'registry.json').read_text())
    assert [r['gate']['passed'] for r in registry][-1] is False


def test_predict_is_logged_idempotently(grid, lake):
    if not (config.MODELS / 'current.json').exists():
        model.promote(*model.train_candidate(grid))
    out = forecast.predict(grid, '2026-06-28')
    assert len(out) == 24 and out['forecast_mw'].between(20000, 110000).all()
    forecast.append(out)
    later = forecast.predict(grid, '2026-06-28')
    later['forecast_mw'] += 1000                  # a later re-run that would look different
    forecast.append(later)
    log = pd.read_parquet(config.FORECASTS)
    day = log[log['issue_date'] == '2026-06-28']
    assert len(day) == 24
    assert day['forecast_mw'].tolist() == out['forecast_mw'].tolist()   # the published one is kept
    forecast.append(later, replace=True)
    assert pd.read_parquet(config.FORECASTS).query("issue_date == '2026-06-28'")['forecast_mw'].tolist() == later['forecast_mw'].tolist()


def test_predict_refuses_missing_features(grid, lake):
    if not (config.MODELS / 'current.json').exists():
        model.promote(*model.train_candidate(grid))
    with pytest.raises(RuntimeError, match='miss features'):
        forecast.predict(grid, '2026-07-10')   # no data or weather that far


def test_evaluate_and_site(grid, lake, tmp_path):
    if not (config.MODELS / 'current.json').exists():
        model.promote(*model.train_candidate(grid))
    forecast.append(forecast.predict(grid, '2026-06-27'))
    ev = forecast.evaluate(grid)
    assert ev['days'] and ev['days'][0]['model'] is not None
    forecast.write_json(ev, config.REPORTS / 'live.json')
    out = site.build(grid, tmp_path / 'public')
    data = json.loads((out / 'data.json').read_text())
    assert data['forecasts'] and data['chart']['hours'] and (out / 'index.html').exists()
