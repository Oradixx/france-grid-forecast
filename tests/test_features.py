import pandas as pd

from gridforecast import features


def test_hourly_index_is_continuous(grid):
    steps = grid.index.to_series().diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(hours=1)]


def test_lags_only_use_data_available_at_issue_time(grid):
    f = features.build(grid)
    t = f.index[500]
    assert f.loc[t, 'lag_48h'] == grid['consumption_mw'].get(t - pd.Timedelta(hours=48))
    assert f.loc[t, 'lag_168h'] == grid['consumption_mw'].get(t - pd.Timedelta(hours=168))


def test_target_hours_follow_paris_days():
    assert len(features.target_hours('2026-03-28')) == 23   # spring forward on 29 March 2026
    assert len(features.target_hours('2026-10-24')) == 25   # fall back on 25 October 2026
    hours = features.target_hours('2026-06-29')
    assert hours[0] == pd.Timestamp('2026-06-29 22:00', tz='UTC')   # midnight in Paris (UTC+2)
    assert len(hours) == 24


def test_calendar_features():
    idx = pd.date_range('2026-05-13 22:00', periods=48, freq='h', tz='UTC')  # 14-15 May 2026 in Paris
    f = features.build(pd.DataFrame({'consumption_mw': 50000.0, 'temperature_c': 15.0}, index=idx))
    day = f.index.tz_convert('Europe/Paris').date
    assert f.loc[day == pd.Timestamp('2026-05-14').date(), 'is_holiday'].eq(1).all()   # Ascension (Thursday)
    assert f.loc[day == pd.Timestamp('2026-05-15').date(), 'is_bridge'].eq(1).all()    # the Friday "pont"
