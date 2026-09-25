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
