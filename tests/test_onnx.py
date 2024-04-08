import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch.nn
from onnxruntime import InferenceSession

from juniper.common.data_type import FeatureType
from juniper.preprocessor.preprocessor import ColumnTransformer
from juniper.training.model_wrapper import TorchModel, Model


@pytest.fixture
def onnx_schema(feature_store):
    """Remove certain column types from the schema until they are ready to be supported"""
    schema = feature_store.get_schema()
    return pa.schema([field for field in schema if field.metadata[b"usable_type"].decode() != FeatureType.TIMESTAMP])


def test_onnx_export(feature_store, onnx_schema):
    column_transformer = ColumnTransformer(feature_store, schema=onnx_schema)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model_onnx = column_transformer.to_onnx()
    assert model_onnx is not None
    with open("testcase.onnx", "wb") as f:
        f.write(model_onnx.SerializeToString())


def test_runtime(feature_store, onnx_schema):
    column_transformer = ColumnTransformer(feature_store, schema=onnx_schema)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model_onnx = column_transformer.to_onnx()
    sess = InferenceSession(model_onnx.SerializeToString())
    input = {}
    for node in sess.get_inputs():
        if node.name.startswith("arr"):
            input[node.name] = np.array([[None, None, None]], dtype=np.float32).reshape(-1, 1)
        elif node.type == "tensor(string)":
            input[node.name] = np.array([[None]], dtype=np.str_)
        else:
            input[node.name] = np.array([[None]], dtype=np.float32)
    output = sess.run(["features", "arr"], input)
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


class SimpleModel(torch.nn.Module, Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inputs = {}
        self.outputs = []
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, x: pd.DataFrame):
        arr = torch.tensor(list(map(lambda y: np.mean(y, axis=1), x["arr"].values)), dtype=torch.float32)
        x_ = torch.tensor(x.drop(columns=["arr"]).values.T, dtype=torch.float32)
        x_ = torch.cat([x_, arr], dim=0)
        return self.linear(x_)


def test_simple_model(feature_store, onnx_schema):
    column_transformer = ColumnTransformer(feature_store, schema=onnx_schema)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model = TorchModel(
        model_cls=SimpleModel,
        loss_fn=lambda: None,
        preprocessor=column_transformer,
    )
    model.fit(pd.DataFrame(), pd.DataFrame(), hyperparameters={"epochs": 0, "learning_rate": 0})
