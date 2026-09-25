"""Paths, sources and constants shared by the pipeline."""
import csv
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]

# The lake lives on the `data` branch, checked out into ./lake by the workflows
LAKE = Path(os.environ.get('GRID_LAKE', ROOT / 'lake'))
RAW_ECO2MIX = LAKE / 'raw' / 'eco2mix'
RAW_WEATHER = LAKE / 'raw' / 'weather'
FORECASTS = LAKE / 'forecasts' / 'forecasts.parquet'
MODELS = LAKE / 'models'
REPORTS = LAKE / 'reports'
WAREHOUSE = Path(os.environ.get('GRID_WAREHOUSE', ROOT / 'warehouse.duckdb'))

PARIS = ZoneInfo('Europe/Paris')

# Open Data Réseaux Énergies (ODRÉ), éCO2mix national data, Licence Ouverte 2.0
ODRE = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets'
ECO2MIX_HISTORY = 'eco2mix-national-cons-def'  # definitive + consolidated, 2012 to ~3 months ago
ECO2MIX_REALTIME = 'eco2mix-national-tr'       # real time, last months, updated every 15 min

# Open-Meteo (free API for non-commercial use, data under CC BY 4.0)
METEO_ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
METEO_FORECAST = 'https://api.open-meteo.com/v1/forecast'

# National temperature = weighted mean of large cities (weights: rough metro-area populations,
# because heating and air conditioning follow where people live). One list for the whole project:
# the dbt seed that the SQL models join on.
CITIES = {row['city']: (float(row['latitude']), float(row['longitude']), float(row['weight']))
          for row in csv.DictReader(open(ROOT / 'dbt' / 'seeds' / 'cities.csv', encoding='utf-8'))}

HISTORY_START = '2012-01-01'
# Train on everything before this date, backtest on what follows
BACKTEST_START = '2025-01-01'
