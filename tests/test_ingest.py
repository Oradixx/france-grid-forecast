import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gridforecast import config, ingest


class FakeResponse:
    def __init__(self, content=b'', payload=None, status=200):
        self.content, self._payload, self.status_code, self.text = content, payload, status, ''

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def parquet_bytes(df):
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


def test_realtime_upsert_keeps_old_rows_and_replaces_revised_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'LAKE', tmp_path)
    monkeypatch.setattr(config, 'RAW_ECO2MIX', tmp_path / 'raw' / 'eco2mix')
    ts = pd.date_range('2026-09-20', periods=4, freq='15min', tz='UTC')
    first = pd.DataFrame({'date_heure': ts, 'consommation': [1, 2, 3, 4]})
    revised = pd.DataFrame({'date_heure': ts[2:].append(pd.DatetimeIndex([ts[-1] + pd.Timedelta('15min')])),
                            'consommation': [30, 40, 5]})
    calls = []

    def fake_get(url, params=None):
        calls.append(params)
        return FakeResponse(parquet_bytes(first if len(calls) == 1 else revised))
    monkeypatch.setattr(ingest, 'get', fake_get)
    ingest.ingest_realtime()
    ingest.ingest_realtime()
    assert calls[0] is None                    # first run: whole dataset
    assert 'date_heure >=' in calls[1]['where']  # then incremental
    out = pd.read_parquet(tmp_path / 'raw/eco2mix/realtime/month=2026-09.parquet')
    assert out['consommation'].tolist() == [1, 2, 30, 40, 5]


def test_retry_on_server_error(monkeypatch):
    responses = [FakeResponse(status=503), FakeResponse(payload={'ok': True})]
    monkeypatch.setattr(ingest.session, 'get', lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(ingest.time, 'sleep', lambda s: None)
    assert ingest.get('https://example.org').json() == {'ok': True}


def test_annotations_are_escaped_for_github(monkeypatch, capsys):
    from gridforecast import gha
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    gha.annotate('error', 'line 1\nline 2: 100%', title='Extract failed: a, b')
    assert capsys.readouterr().out.strip() == '::error title=Extract failed%3A a%2C b::line 1%0Aline 2: 100%25'


def test_client_errors_are_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(ingest.session, 'get', lambda *a, **k: calls.append(1) or FakeResponse(status=400))
    monkeypatch.setattr(ingest.time, 'sleep', lambda s: None)
    try:
        ingest.get('https://example.org', {'where': 'x'})
    except ingest.ClientError as e:
        assert '400 for https://example.org' in str(e)
    assert len(calls) == 1


def test_run_tries_every_source_and_reports_each_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setattr(config, 'RAW_ECO2MIX', tmp_path / 'raw' / 'eco2mix')
    (tmp_path / 'raw' / 'eco2mix' / 'history').mkdir(parents=True)
    (tmp_path / 'raw' / 'eco2mix' / 'history' / 'year=2026.parquet').touch()
    done = []

    def boom():
        raise ingest.ClientError('400 for odre: bad where')
    monkeypatch.setattr(ingest, 'ingest_realtime', boom)
    monkeypatch.setattr(ingest, 'weather_archive_start', lambda: None)
    monkeypatch.setattr(ingest, 'ingest_weather_archive', lambda start=None: done.append('archive'))
    monkeypatch.setattr(ingest, 'ingest_weather_forecast', lambda: done.append('forecast'))
    try:
        ingest.run()
        raise AssertionError('should fail')
    except ingest.IngestError as e:
        assert 'real time' in str(e)
    assert done == ['archive', 'forecast']
    # ':' is escaped in the title (a property), not in the message
    assert '::error title=Extract failed%3A éCO2mix real time (ODRÉ)::ClientError: 400 for odre: bad where' \
        in capsys.readouterr().out.splitlines()


def test_freshness_warns_when_measures_stop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setattr(config, 'LAKE', tmp_path)
    monkeypatch.setattr(config, 'RAW_ECO2MIX', tmp_path / 'raw' / 'eco2mix')
    ts = pd.date_range('2026-10-05 08:00', periods=12, freq='h', tz='UTC')
    df = pd.DataFrame({'date_heure': ts, 'consommation': [50000.0] * 2 + [None] * 10})  # forecast-only rows after
    ingest.write_partition(df, config.RAW_ECO2MIX / 'realtime' / 'month=2026-10.parquet')
    lag = ingest.realtime_freshness(now=pd.Timestamp('2026-10-07 10:00', tz='UTC'))
    assert round(lag) == 49
    assert '::warning title=RTE real-time data is late::last measured consumption is 2026-10-05 09:00' \
        in capsys.readouterr().out
