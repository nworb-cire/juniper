import pyarrow as pa
import pytest
import torch

from juniper.common.data_type import FeatureType
from juniper.preprocessor.preprocessor import ColumnTransformer
from juniper.modeling.layers import Unify, SummaryPool


@pytest.fixture
def onnx_schema(feature_store):
    """Remove certain column types from the schema until they are ready to be supported"""
    schema = feature_store.get_schema()
    return pa.schema([field for field in schema if field.metadata[b"usable_type"].decode() != FeatureType.TIMESTAMP])


def test_unify(onnx_schema, data):
    column_transformer = ColumnTransformer(schema=onnx_schema)
    x = column_transformer.fit_transform(data)
    unify = Unify({"arr": SummaryPool()})
    y = unify(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] - 1 + 4 * len(x["arr"].iloc[0])
