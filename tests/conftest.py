import os

import pytest

from juniper.data_loading.feature_store import LocalParquetFeatureStore


@pytest.fixture
def config():
    os.environ["CONFIG_LOCATION"] = "tests/test_config.toml"
    yield
    del os.environ["CONFIG_LOCATION"]


@pytest.fixture
def feature_store(config):
    return LocalParquetFeatureStore()
