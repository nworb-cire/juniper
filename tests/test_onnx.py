import numpy as np
import pyarrow as pa
import pytest
from onnxruntime import InferenceSession

from juniper.common.data_type import FeatureType
from juniper.common.export import to_onnx
from juniper.preprocessor.preprocessor import get_preprocessor


@pytest.fixture
def onnx_schema(feature_store):
    """Remove certain column types from the schema until they are ready to be supported"""
    schema = feature_store.get_schema()
    return pa.schema([field for field in schema if field.metadata[b"usable_type"].decode() != FeatureType.TIMESTAMP])


def test_onnx_export(feature_store, onnx_schema):
    column_transformer = get_preprocessor(feature_store, schema=onnx_schema)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model_onnx = to_onnx(column_transformer, "test")
    assert model_onnx is not None
    with open("testcase.onnx", "wb") as f:
        f.write(model_onnx.SerializeToString())


def test_runtime(feature_store, onnx_schema):
    column_transformer = get_preprocessor(feature_store, schema=onnx_schema)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model_onnx = to_onnx(column_transformer, "test")
    sess = InferenceSession(model_onnx.SerializeToString())
    input = {}
    for node in sess.get_inputs():
        if node.name.startswith("arr"):
            input[node.name] = np.array([[None, None, None]], dtype=np.float32).reshape(-1, 1)
        elif node.type == "tensor(string)":
            input[node.name] = np.array([[None]], dtype=np.str_)
        else:
            input[node.name] = np.array([[None]], dtype=np.float32)
    output = sess.run(["test_output", "test_arr_output"], input)
    assert output is not None
    assert len(output) == 2
    expected = np.array([0.0, 3.0, -1.0])
    assert np.allclose(output[0], expected)
    expected = np.array(
        [
            [-0.6377551, -0.7653061],
            [-0.6377551, -0.7653061],
            [-0.6377551, -0.7653061],
        ]
    )
    assert np.allclose(output[1], expected)
