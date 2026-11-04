import json
import shutil

import numpy as np
import pandas as pd
import pytest

from gridforecast import cli, config, features, forecast, intervals, model, site

START = '2026-03-01'


@pytest.fixture(scope='session')
def interval_backtest(grid):
    report, test = model.backtest(grid, start=START)
    return intervals.backtest(grid, test, start=START), test


@pytest.fixture(scope='session')
def trained(grid):
    return intervals.train(grid)


def point_model_only(grid, d):
    """A models directory holding the current point model and no interval models."""
    if not (config.MODELS / 'current.json').exists():
        model.promote(*model.train_candidate(grid))
    version = json.loads((config.MODELS / 'current.json').read_text())['version']
    d.mkdir()
    for name in ('current.json', f'{version}.joblib'):
        shutil.copy(config.MODELS / name, d / name)
    return d


def test_coverage_and_pinball_by_hand():
    y = np.array([100., 100., 100., 100.])
    lo = np.array([90., 101., 95., 80.])     # the second hour is below P10
    hi = np.array([110., 120., 99., 120.])   # the third is above P90
    s = intervals.scores(y, lo, hi)
    assert (s['coverage_pct'], s['below_p10_pct'], s['above_p90_pct']) == (50, 25, 25)
    # y - P10 = 10, -1, 5, 20: 0.1 per MW above the bound, 0.9 per MW below it
    assert s['pinball_p10_mw'] == pytest.approx((1 + 0.9 + 0.5 + 2) / 4)
    # y - P90 = -10, -20, 1, -20: 0.1 per MW below the bound, 0.9 per MW above it
    assert s['pinball_p90_mw'] == pytest.approx((1 + 2 + 0.9 + 2) / 4)
    # for the median, the pinball loss is half the absolute error
    assert intervals.pinball(y, [90, 110, 100, 105], 0.5) == pytest.approx(np.mean([10, 10, 0, 5]) / 2)


def test_calibration_leaves_one_hour_in_ten_beyond_the_bound():
    beyond = np.random.default_rng(1).normal(size=1000)   # how far each hour is beyond the bound
    move = intervals._move(beyond, 0.1)
    assert 0.09 < np.mean(beyond > move) <= 0.1


def test_coverage_interval_accounts_for_misses_in_runs():
    hours = pd.date_range('2025-01-01', periods=24 * 7 * 30, freq='h', tz='UTC')
    # whole days missed, as in a cold spell: 210 days, not 5040 independent hours
    missed_days = np.random.default_rng(0).random(len(hours) // 24) < 0.2
    inside = ~np.repeat(missed_days, 24)
    lo, hi = intervals._coverage_ci(inside, hours)
    p = inside.mean()
    binomial = 1.96 * np.sqrt(p * (1 - p) / len(inside)) * 100
    assert lo < p * 100 < hi
    assert (hi - lo) / 2 > 2 * binomial


def test_backtest_reports_coverage_and_pinball_on_the_point_backtest_hours(interval_backtest):
    iv, test = interval_backtest
    m, naive = iv['overall']['model'], iv['overall']['naive_band']
    assert m['hours'] == len(test)
    assert m['coverage_pct'] + m['below_p10_pct'] + m['above_p90_pct'] == pytest.approx(100)
    assert m['coverage_ci95_pct'][0] <= m['coverage_pct'] <= m['coverage_ci95_pct'][1]
    assert 50 < m['coverage_pct'] < 98
    assert m['mean_width_mw'] > 0
    # a band that cannot beat "last week, give or take the usual change" would be useless
    assert m['pinball_mw'] < naive['pinball_mw']
    assert {'constant_band', 'naive_band'} <= set(iv['overall'])
    assert [x['month'] for x in iv['monthly']] == sorted({t.strftime('%Y-%m') for t in test.index.tz_convert(config.PARIS)})
    assert iv['signature'] == intervals.code_signature()


def test_backtest_calibrates_on_training_data_only(interval_backtest):
    cal = interval_backtest[0]['calibration']
    assert pd.Timestamp(cal['until']) < pd.Timestamp(START, tz='UTC')
    assert cal['hours'] > 0 and cal['uncalibrated']['coverage_pct'] is not None


def test_interval_models_share_the_point_model_features_and_settings(trained):
    im, meta = trained
    for bound, q in zip(im.bounds, intervals.QUANTILES):
        params = bound.est.get_params()
        assert list(bound.est.feature_names_in_) == features.FEATURES
        assert (params['loss'], params['quantile']) == ('quantile', q)
        assert {k: params[k] for k in model.PARAMS} == model.PARAMS
    assert meta['version'].endswith(intervals.code_signature())


def test_a_change_of_the_point_model_code_retrains_the_interval_models(monkeypatch):
    before = intervals.code_signature()
    monkeypatch.setattr(model, 'PARAMS', {**model.PARAMS, 'max_iter': 1})
    assert intervals.code_signature() != before


def test_predict_adds_the_band_and_leaves_the_point_forecast_alone(grid, lake, trained, tmp_path):
    d = point_model_only(grid, tmp_path / 'models')
    without = forecast.predict(grid, '2026-06-28', models_dir=d)
    assert without['p10_mw'].isna().all() and without['interval_version'].isna().all()   # published anyway
    intervals.save(*trained, models_dir=d)
    out = forecast.predict(grid, '2026-06-28', models_dir=d)
    assert out['forecast_mw'].tolist() == without['forecast_mw'].tolist()
    assert (out['p10_mw'] <= out['p90_mw']).all() and out['p10_mw'].between(20000, 110000).all()
    assert out['interval_version'].eq(trained[1]['version']).all()


def test_interval_models_from_other_code_are_not_used(grid, lake, trained, tmp_path):
    d = point_model_only(grid, tmp_path / 'models')
    im, meta = trained
    intervals.save(im, {**meta, 'version': '20260630-0ther0'}, models_dir=d)
    assert intervals.is_stale(d)
    with pytest.raises(RuntimeError, match='other code'):
        intervals.load_current(d)
    assert forecast.predict(grid, '2026-06-28', models_dir=d)['p10_mw'].isna().all()


def test_a_failing_interval_model_does_not_stop_the_forecast(grid, lake, trained, tmp_path, monkeypatch):
    d = point_model_only(grid, tmp_path / 'models')
    intervals.save(*trained, models_dir=d)

    def broken(self, X):
        raise AttributeError('saved by an older version of the class')
    monkeypatch.setattr(intervals.IntervalModel, 'predict', broken)
    out = forecast.predict(grid, '2026-06-28', models_dir=d)
    assert out['forecast_mw'].notna().all() and out['p10_mw'].isna().all()


def test_a_log_from_before_the_interval_still_works(grid, lake, trained, tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'FORECASTS', tmp_path / 'forecasts.parquet')
    d = point_model_only(grid, tmp_path / 'models')
    intervals.save(*trained, models_dir=d)
    # the log on the data branch has no interval columns
    forecast.append(forecast.predict(grid, '2026-06-27', models_dir=d).drop(columns=['p10_mw', 'p90_mw', 'interval_version']))
    forecast.append(forecast.predict(grid, '2026-06-28', models_dir=d))
    forecast.evaluate(grid)   # rewrites the log
    log = pd.read_parquet(config.FORECASTS)
    assert log.loc[log['issue_date'] == '2026-06-27', 'p10_mw'].isna().all()
    assert log.loc[log['issue_date'] == '2026-06-28', 'p10_mw'].notna().all()
    data = json.loads((site.build(grid, tmp_path / 'public') / 'data.json').read_text())
    p10 = {f['issue_date']: f['p10'] for f in data['forecasts']}
    assert p10['2026-06-27'] is None and p10['2026-06-28'] is not None


def test_cli_reruns_what_the_interval_needs(grid, lake, trained, interval_backtest, monkeypatch, capsys):
    iv, test = interval_backtest
    monkeypatch.setattr(model, 'backtest', lambda df, start: ({'model_signature': model.code_signature(), 'overall': {}}, test))
    monkeypatch.setattr(intervals, 'backtest', lambda df, test, start: iv)
    monkeypatch.setattr(intervals, 'train', lambda df: trained)
    # a report written before the interval existed, by the same point model code: not up to date
    rep = config.REPORTS / 'backtest.json'
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({'model_signature': model.code_signature()}))
    assert cli.main(['backtest', '--if-stale', '--start', START]) == 0
    assert json.loads(rep.read_text())['intervals']['signature'] == intervals.code_signature()
    assert cli.main(['train-intervals', '--if-stale']) == 0
    assert not intervals.is_stale()
    capsys.readouterr()
    cli.main(['backtest', '--if-stale', '--start', START])
    cli.main(['train-intervals', '--if-stale'])
    out = capsys.readouterr().out
    assert 'backtest report matches the code, skipped' in out and 'interval models match the code' in out
