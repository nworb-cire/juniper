import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import parquet as pq


@pytest.fixture
def data_path() -> str:
    return "data/feature_store.parquet"


@pytest.fixture
def data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


@pytest.fixture
def schema(data_path: str) -> pa.Schema:
    return pq.read_schema(data_path)
