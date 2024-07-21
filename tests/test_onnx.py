import tempfile

import numpy as np
import onnx
import pandas as pd
import pyarrow as pa
import pytest
import torch.nn
from onnxruntime import InferenceSession

from juniper.common.data_type import FeatureType
from juniper.preprocessor.preprocessor import ColumnTransformer
from juniper.modeling.model_wrapper import Model
from juniper.modeling.torch import TorchModel


def test_onnx_export(onnx_schema, data):
    column_transformer = ColumnTransformer(schema=onnx_schema)
    column_transformer.fit(data)

    model_onnx = column_transformer.to_onnx()
    assert model_onnx is not None

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(model_onnx.SerializeToString())
        temp.flush()
        model_onnx = onnx.load_model(temp.name)
        assert model_onnx is not None


def test_runtime(onnx_schema, data):
    column_transformer = ColumnTransformer(schema=onnx_schema)
    column_transformer.fit(data)

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


def test_simple_model(onnx_schema, data):
    column_transformer = ColumnTransformer(schema=onnx_schema)
    column_transformer.fit(data)

    model = TorchModel(
        model_cls=SimpleModel,
        loss_fn=lambda: None,
        preprocessor=column_transformer,
    )
    model.fit(pd.DataFrame(), pd.DataFrame(), hyperparameters={"epochs": 0, "learning_rate": 0})
