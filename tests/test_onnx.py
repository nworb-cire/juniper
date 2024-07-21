import tempfile

import numpy as np
import onnx
from onnxruntime import InferenceSession

from juniper.preprocessor.preprocessor import ColumnTransformer


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
