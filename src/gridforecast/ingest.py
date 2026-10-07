"""Extract: download éCO2mix and weather data into the raw layer of the lake (Parquet).

Raw files are stored as downloaded (all columns, original types), partitioned so a daily run
only rewrites what changed:

    lake/raw/eco2mix/history/year=2019.parquet     definitive/consolidated, refreshed monthly
    lake/raw/eco2mix/realtime/month=2026-09.parquet real time, the last days re-downloaded every run
    lake/raw/weather/archive/year=2019.parquet      observed temperatures (reanalysis)
    lake/raw/weather/forecast/issued=2026-09-26.parquet  the forecast used for each prediction

Every write is idempotent: a partition is replaced as a whole, never appended to, so a run can be
retried safely.
"""
import io
import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from . import config, gha

log = logging.getLogger(__name__)
session = requests.Session()
session.headers['User-Agent'] = 'france-grid-forecast (github.com/Oradixx/france-grid-forecast)'


class ClientError(Exception):
    """4xx answer (other than 429): not retried."""


def get(url, params=None, retries=4):
    """GET with retries and exponential backoff: public APIs fail transiently."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=300)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f'{r.status_code} {r.text[:200]}')
            if r.status_code >= 400:
                # a client error will not fix itself: fail now, with the API's own explanation
                raise ClientError(f'{r.status_code} for {url} {params or ""}: {r.text[:300]}')
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == retries - 1:
                raise
            wait = (60 if '429' in str(e) else 5) * 2 ** attempt
            log.warning('%s failed (%s), retry in %ss', url, e, wait)
            time.sleep(wait)


def write_partition(df, path):
    """Replace one partition atomically (write to a temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression='zstd')
    tmp.replace(path)
    log.info('wrote %s (%d rows)', path.relative_to(config.LAKE), len(df))


# ---------------------------------------------------------------- éCO2mix
def eco2mix_export(dataset, since=None):
    """Download a dataset (or the rows since a date) with the ODRÉ Parquet export."""
    params = {'where': f'date_heure >= "{since}"'} if since else None
    r = get(f'{config.ODRE}/{dataset}/exports/parquet', params)
    df = pq.read_table(io.BytesIO(r.content)).to_pandas()
    log.info('%s: %d rows since %s', dataset, len(df), since or 'the start')
    return df


def ingest_history():
    """Definitive and consolidated data, one file per year. About 25 MB for 2012 to now."""
    df = eco2mix_export(config.ECO2MIX_HISTORY)
    df['date_heure'] = pd.to_datetime(df['date_heure'], utc=True)
    for year, part in df.groupby(df['date_heure'].dt.year):
        write_partition(part.sort_values('date_heure'), config.RAW_ECO2MIX / 'history' / f'year={year}.parquet')


def ingest_realtime(days=4):
    """Re-download real-time data from a few days before the last stored row: RTE revises recent
    values, and missed runs are caught up automatically. First run: the whole dataset."""
    files = sorted((config.RAW_ECO2MIX / 'realtime').glob('month=*.parquet'))
    since = None
    if files:
        last = pd.to_datetime(pd.read_parquet(files[-1], columns=['date_heure'])['date_heure'], utc=True)
        last = last[last.notna()].max()
        since = min(last - timedelta(days=2), datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    new = eco2mix_export(config.ECO2MIX_REALTIME, since=since)
    new['date_heure'] = pd.to_datetime(new['date_heure'], utc=True)
    for month, part in new.groupby(new['date_heure'].dt.strftime('%Y-%m')):
        path = config.RAW_ECO2MIX / 'realtime' / f'month={month}.parquet'
        if path.exists():
            old = pd.read_parquet(path)
            old['date_heure'] = pd.to_datetime(old['date_heure'], utc=True)
            # upsert: new rows win on the timestamp
            part = pd.concat([old[~old['date_heure'].isin(part['date_heure'])], part])
        write_partition(part.sort_values('date_heure'), path)


# ---------------------------------------------------------------- weather
def _meteo_frame(payload):
    """Open-Meteo returns one object per location when several are requested."""
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for city, loc in zip(config.CITIES, payload):
        h = loc['hourly']
        frames.append(pd.DataFrame({'time_utc': pd.to_datetime(h['time'], utc=True), 'city': city,
                                    'temperature': h['temperature_2m']}))
    return pd.concat(frames, ignore_index=True)


def _coords():
    lat = ','.join(str(v[0]) for v in config.CITIES.values())
    lon = ','.join(str(v[1]) for v in config.CITIES.values())
    return lat, lon


def ingest_weather_archive(start=None):
    """Observed hourly temperatures, one file per year. The archive lags by a few days."""
    lat, lon = _coords()
    start = date.fromisoformat(start or config.HISTORY_START)
    end = date.today() - timedelta(days=2)
    for year in range(start.year, end.year + 1):
        a, b = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        payload = get(config.METEO_ARCHIVE, {'latitude': lat, 'longitude': lon, 'start_date': a.isoformat(),
                                             'end_date': b.isoformat(), 'hourly': 'temperature_2m',
                                             'timezone': 'GMT'}).json()
        df = _meteo_frame(payload).dropna(subset=['temperature'])
        path = config.RAW_WEATHER / 'archive' / f'year={year}.parquet'
        if path.exists() and a > date(year, 1, 1):
            old = pd.read_parquet(path)
            old['time_utc'] = pd.to_datetime(old['time_utc'], utc=True)
            df = pd.concat([old[old['time_utc'] < pd.Timestamp(a, tz='UTC')], df])
        write_partition(df.sort_values(['time_utc', 'city']), path)
        # Open-Meteo weights a request by its length and number of locations (a year for 8
        # cities is ~200 "calls" against a limit of 600 per minute): pace the backfill
        if (b - a).days > 31:
            time.sleep(35)


def ingest_weather_forecast(issued=None):
    """Hourly forecast for the next days (plus the last days of model analysis), kept per issue
    date: the forecast used for a prediction must stay what it was at the time."""
    lat, lon = _coords()
    payload = get(config.METEO_FORECAST, {'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m',
                                          'past_days': 7, 'forecast_days': 3, 'timezone': 'GMT'}).json()
    issued = issued or datetime.now(timezone.utc).date().isoformat()
    df = _meteo_frame(payload).dropna(subset=['temperature'])
    df['issued'] = issued
    write_partition(df, config.RAW_WEATHER / 'forecast' / f'issued={issued}.parquet')


def weather_archive_start():
    """Resume the archive where it stops (minus a few days, the archive revises recent values)."""
    files = sorted((config.RAW_WEATHER / 'archive').glob('year=*.parquet'))
    if not files:
        return None
    last = pd.to_datetime(pd.read_parquet(files[-1], columns=['time_utc'])['time_utc']).max()
    return (last - timedelta(days=5)).date().isoformat()


def realtime_freshness(now=None, max_lag_hours=6):
    """Hours since the last measured consumption in the real-time data. RTE publishes its day-ahead
    forecast before the measures, so rows with a forecast but no consumption are ignored."""
    files = sorted((config.RAW_ECO2MIX / 'realtime').glob('month=*.parquet'))
    if not files:
        return None
    df = pd.concat(pd.read_parquet(f, columns=['date_heure', 'consommation']) for f in files[-2:])
    measured = pd.to_datetime(df.loc[df['consommation'].notna(), 'date_heure'], utc=True)
    if measured.empty:
        return None
    last = measured.max()
    lag = ((now or pd.Timestamp.now(tz='UTC')) - last) / pd.Timedelta(hours=1)
    if lag > max_lag_hours:
        gha.annotate('warning', f'last measured consumption is {last:%Y-%m-%d %H:%M} UTC ({lag:.0f} h ago): '
                     'the forecast needs measures up to 48 h before each target hour',
                     title='RTE real-time data is late')
    return lag


class IngestError(Exception):
    pass


def run(full_history=False):
    """Daily: real time + recent weather. `full_history` (weekly) re-downloads the éCO2mix history,
    which RTE consolidates month after month. The weather archive is only backfilled once.

    Each source is tried even if another one fails (they are independent), every failure is
    reported as an annotation, and the step fails at the end if anything failed."""
    steps = []
    if full_history or not any((config.RAW_ECO2MIX / 'history').glob('*.parquet')):
        steps.append(('éCO2mix history (ODRÉ)', ingest_history))
    steps += [('éCO2mix real time (ODRÉ)', ingest_realtime),
              ('weather archive (Open-Meteo)', lambda: ingest_weather_archive(start=weather_archive_start())),
              ('weather forecast (Open-Meteo)', ingest_weather_forecast)]
    failed = []
    for name, step in steps:
        try:
            step()
        except Exception as e:  # noqa: BLE001 - reported, then re-raised as a whole below
            failed.append(name)
            gha.annotate('error', f'{type(e).__name__}: {e}', title=f'Extract failed: {name}')
    realtime_freshness()
    if failed:
        raise IngestError(f'{len(failed)} source(s) failed: {", ".join(failed)}')
