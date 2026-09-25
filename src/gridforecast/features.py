"""Features for the day-ahead forecast. One function for training, backtest and prediction, so
the model never sees features computed differently in production (no train/serve skew).

The forecast is issued in the morning of day D for the 24 hours of day D+1 (Paris time), like
RTE's own day-ahead forecast. At that moment the last complete day is D-1, so a lag can only
look 48 hours back or more: using "the same hour yesterday" (24 h) would leak data that does
not exist yet at issue time.
"""
from functools import lru_cache

import duckdb
import holidays
import numpy as np
import pandas as pd

from . import config

FEATURES = [
    'hour', 'dow', 'doy_sin', 'doy_cos', 'is_holiday', 'is_bridge', 'is_weekend_or_holiday',
    'temperature_c', 'temp_day_mean', 'temp_smooth', 'heating_degrees', 'cooling_degrees',
    'lag_48h', 'lag_168h', 'lag_48h_was_off', 'lag_168h_was_off',
]
TARGET = 'consumption_mw'


def load(warehouse=None):
    """The hourly grid table and the temperature table, on one continuous UTC hourly index that
    extends to the end of the latest temperature forecast (the hours to predict)."""
    con = duckdb.connect(str(warehouse or config.WAREHOUSE), read_only=True)
    grid = con.sql('select * from fct_grid_hourly').df()
    temp = con.sql('select hour_utc, temperature_c from fct_temperature_hourly').df()
    con.close()
    for d in (grid, temp):
        d['hour_utc'] = pd.to_datetime(d['hour_utc']).dt.tz_localize('UTC')
    grid = grid.drop(columns=['temperature_c']).set_index('hour_utc')
    temp = temp.set_index('hour_utc')
    idx = pd.date_range(min(grid.index.min(), temp.index.min()), max(grid.index.max(), temp.index.max()),
                        freq='h', tz='UTC')
    df = grid.reindex(idx).join(temp)
    df.index.name = 'hour_utc'
    return df


@lru_cache(maxsize=None)
def _holidays(years):
    return holidays.France(years=list(years))


def build(df):
    """Add the feature columns to an hourly frame indexed by UTC hour (continuous index)."""
    out = df.copy()
    local = out.index.tz_convert(config.PARIS)
    day = pd.Series(local.date, index=out.index)
    hol = _holidays(tuple(range(local.year.min() - 1, local.year.max() + 2)))

    out['hour'] = local.hour
    out['dow'] = local.dayofweek
    out['doy_sin'] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    out['is_holiday'] = day.map(lambda d: d in hol).astype(int)
    # "pont": a Monday before a Tuesday holiday or a Friday after a Thursday holiday
    prev_day = day.map(lambda d: (d - pd.Timedelta(days=1)) in hol)
    next_day = day.map(lambda d: (d + pd.Timedelta(days=1)) in hol)
    out['is_bridge'] = (((out['dow'] == 0) & next_day) | ((out['dow'] == 4) & prev_day)).astype(int)
    out['is_weekend_or_holiday'] = ((out['dow'] >= 5) | (out['is_holiday'] == 1)).astype(int)

    # Temperature: the hour, the mean of the local day, and a slow average (buildings take days
    # to cool down or warm up, so heating follows the last days' weather, not the current hour)
    t = out['temperature_c']
    out['temp_day_mean'] = t.groupby(day).transform('mean')
    out['temp_smooth'] = t.ewm(halflife=48, ignore_na=True).mean()
    out['heating_degrees'] = (15.0 - out['temp_smooth']).clip(lower=0)
    out['cooling_degrees'] = (out['temperature_c'] - 22.0).clip(lower=0)

    # Lags, available at issue time (see module docstring)
    out['lag_48h'] = out[TARGET].shift(48)
    out['lag_168h'] = out[TARGET].shift(168)
    out['lag_48h_was_off'] = out['is_weekend_or_holiday'].shift(48)
    out['lag_168h_was_off'] = out['is_weekend_or_holiday'].shift(168)
    return out


def target_hours(issue_date):
    """UTC hours of the local day after `issue_date` (23 or 25 hours on DST change days)."""
    day = pd.Timestamp(issue_date) + pd.Timedelta(days=1)
    start = pd.Timestamp(day.date(), tz=config.PARIS)
    end = pd.Timestamp((day + pd.Timedelta(days=1)).date(), tz=config.PARIS)
    return pd.date_range(start.tz_convert('UTC'), end.tz_convert('UTC'), freq='h', inclusive='left')
