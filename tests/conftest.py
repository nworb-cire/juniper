import os
from pathlib import Path

import pandas as pd
import pytest

from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.data_loading.outcomes import StandardOutcomes


@pytest.fixture
def config():
    os.environ["CONFIG_LOCATION"] = "tests/test_config.toml"
    yield
    del os.environ["CONFIG_LOCATION"]


@pytest.fixture
def feature_store(config):
    return LocalParquetFeatureStore()


class TestOutcomes(StandardOutcomes):
    def _get_columns(self, df: pd.DataFrame) -> list[str]:
        return list(self.binary_outcomes_list)

    def _path_str(self) -> str:
        return "test"

    def read_parquet(
        self, path: Path = None, columns: list[str] = None, filters: list[tuple] | list[list[tuple]] | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.index_column: range(365),
                self.timestamp_column: pd.date_range(start="2021-01-01", periods=365, freq="D"),
                "outcome": [0] * 365,
            }
        )


@pytest.fixture
def outcomes(config):
    return TestOutcomes()
