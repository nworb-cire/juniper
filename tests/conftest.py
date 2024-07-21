import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import parquet as pq

from juniper.common.data_type import FeatureType


@pytest.fixture(scope="session")
def data_path() -> str:
    # path of current file
    current_filepath = Path(os.path.abspath(__file__))
    return (current_filepath.parent / "data/feature_store.parquet").absolute().as_posix()


@pytest.fixture
def data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


@pytest.fixture
def schema(data_path: str) -> pa.Schema:
    return pq.read_schema(data_path)


@pytest.fixture
def onnx_schema(schema):
    """Remove certain column types from the schema until they are ready to be supported"""
    return pa.schema([field for field in schema if field.metadata[b"usable_type"].decode() != FeatureType.TIMESTAMP])
