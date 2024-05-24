import os

import pytest

from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.data_loading.outcomes import LocalStandardOutcomes


@pytest.fixture
def config():
    os.environ["CONFIG_LOCATION"] = "tests/test_config.toml"
    yield
    del os.environ["CONFIG_LOCATION"]


@pytest.fixture
def feature_store(config):
    return LocalParquetFeatureStore()


class TestOutcomes(LocalStandardOutcomes):
    def _get_columns(self, columns: list[str] = None) -> list[str]:
        return list(self.binary_outcomes_list)

    def _path_str(self) -> str:
        return "test"


@pytest.fixture
def outcomes(config):
    return TestOutcomes()
