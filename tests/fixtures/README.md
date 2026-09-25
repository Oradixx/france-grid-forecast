# Test fixtures

`lake/` is a small lake the tests run the whole pipeline on, without network access. It was built by
[`build_fixtures.py`](build_fixtures.py) from a real ODRÉ export:

- éCO2mix history: January 2025 to June 2026, real data (Licence Ouverte 2.0), including two
  daylight-saving changes of each kind.
- éCO2mix real time: the last ten days of that history, in the real-time schema, overlapping the
  history by five days with values shifted by +1 % (staging must keep the history version).
- Weather: synthetic temperatures.
