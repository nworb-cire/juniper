import os
from pathlib import Path

import pandas as pd
import pyarrow as pa

import pytest
from pyarrow import parquet as pq

from juniper.data_loading.feature_store import ParquetFeatureStore


@pytest.fixture
def config():
    os.environ["CONFIG_LOCATION"] = "tests/test_config.toml"
    yield
    del os.environ["CONFIG_LOCATION"]


class FeatureStoreTest(ParquetFeatureStore):
    def get_schema(self) -> pa.Schema:
        return pq.read_schema(self.path)

    def read_parquet(
        self, path: Path = None, columns: list[str] = None, filters: list[tuple] | list[list[tuple]] | None = None
    ) -> pd.DataFrame:
        df = pd.read_parquet(self.path, columns=columns, filters=filters)
        return df.set_index(self.index_column)


@pytest.fixture
def feature_store(config):
    return FeatureStoreTest()
