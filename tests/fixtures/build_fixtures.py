"""Build the small lake that CI runs the pipeline on (no network in tests).

- eco2mix history: real ODRÉ export, January 2025 to June 2026 (Licence Ouverte 2.0).
- eco2mix real time: the last ten days of that history, relabelled as real-time rows in the
  real-time schema (which has extra columns and other types), overlapping the history by five
  days with values shifted by +1 %: staging must keep the history version.
- weather: synthetic temperatures (seasonal + daily cycle), the tests do not need real ones.

Usage: python tests/fixtures/build_fixtures.py <def_export.parquet>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAKE = HERE / 'lake'
src = pd.read_parquet(sys.argv[1])
src['date_heure'] = pd.to_datetime(src['date_heure'], utc=True)
hist = src[(src['date_heure'] >= '2025-01-01') & (src['date_heure'] < '2026-06-26')]
(LAKE / 'raw/eco2mix/history').mkdir(parents=True, exist_ok=True)
for year, part in hist.groupby(hist['date_heure'].dt.year):
    part.to_parquet(LAKE / f'raw/eco2mix/history/year={year}.parquet', index=False, compression='zstd')

rt = src[src['date_heure'] >= '2026-06-21'].copy()
rt['nature'] = 'Données temps réel'
overlap = rt['date_heure'] < '2026-06-26'
for col in ['consommation', 'prevision_j1']:
    rt.loc[overlap, col] = (rt.loc[overlap, col] * 1.01).round()
rt['ech_comm_allemagne_belgique'] = pd.to_numeric(rt['ech_comm_allemagne_belgique'], errors='coerce').astype('Int64')
rt['eolien_terrestre'] = rt['eolien']
rt['eolien_offshore'] = 0
rt['stockage_batterie'] = pd.NA
(LAKE / 'raw/eco2mix/realtime').mkdir(parents=True, exist_ok=True)
rt.to_parquet(LAKE / 'raw/eco2mix/realtime/month=2026-06.parquet', index=False, compression='zstd')

cities = pd.read_csv(HERE.parents[1] / 'dbt/seeds/cities.csv')
hours = pd.date_range('2024-12-31', '2026-07-03', freq='h', tz='UTC')
doy = hours.dayofyear.to_numpy()
hod = hours.hour.to_numpy()
rng = np.random.default_rng(0)
frames = []
for i, c in cities.iterrows():
    t = 12 - 8 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 1.5, len(hours)) - i * 0.3
    frames.append(pd.DataFrame({'time_utc': hours, 'city': c['city'], 'temperature': t.round(1)}))
w = pd.concat(frames)
obs = w[w['time_utc'] < '2026-06-29']
(LAKE / 'raw/weather/archive').mkdir(parents=True, exist_ok=True)
for year, part in obs.groupby(obs['time_utc'].dt.year):
    part.to_parquet(LAKE / f'raw/weather/archive/year={year}.parquet', index=False)
fc = w[w['time_utc'] >= '2026-06-24'].copy()
fc['issued'] = '2026-06-30'
(LAKE / 'raw/weather/forecast').mkdir(parents=True, exist_ok=True)
fc.to_parquet(LAKE / 'raw/weather/forecast/issued=2026-06-30.parquet', index=False)
print('fixture lake written to', LAKE)
