import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_LAKE = ROOT / 'tests' / 'fixtures' / 'lake'


@pytest.fixture(scope='session')
def lake(tmp_path_factory, request):
    """A copy of the fixture lake (the pipeline writes into it) and a warehouse built by dbt."""
    base = tmp_path_factory.mktemp('run')
    lake = base / 'lake'
    shutil.copytree(FIXTURE_LAKE, lake)
    warehouse = base / 'warehouse.duckdb'
    env = {'GRID_LAKE': str(lake), 'GRID_WAREHOUSE': str(warehouse)}
    mp = pytest.MonkeyPatch()
    for k, v in env.items():
        mp.setenv(k, v)
    from gridforecast import config
    mp.setattr(config, 'LAKE', lake)
    mp.setattr(config, 'WAREHOUSE', warehouse)
    mp.setattr(config, 'RAW_ECO2MIX', lake / 'raw' / 'eco2mix')
    mp.setattr(config, 'RAW_WEATHER', lake / 'raw' / 'weather')
    mp.setattr(config, 'FORECASTS', lake / 'forecasts' / 'forecasts.parquet')
    mp.setattr(config, 'MODELS', lake / 'models')
    mp.setattr(config, 'REPORTS', lake / 'reports')
    subprocess.run(['dbt', 'build', '--profiles-dir', '.', '--target-path', str(base / 'dbt-target'),
                    '--vars', '{check_freshness: false}'], cwd=ROOT / 'dbt', check=True,
                   env={**__import__('os').environ, **env})
    request.addfinalizer(mp.undo)
    return lake


@pytest.fixture(scope='session')
def grid(lake):
    from gridforecast import features
    return features.load()
