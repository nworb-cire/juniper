import importlib
import logging

import numpy as np
import onnx
import onnxruntime
import pandas as pd

from juniper.common.setup import load_config


def to_human_readable_number(x: int) -> str:
    if x < 1e3:
        return str(x)
    elif x < 1e6:
        return f"{x/1e3:.1f}k"
    elif x < 1e9:
        return f"{x/1e6:.1f}M"
    else:
        return f"{x/1e9:.1f}B"


# TODO: Use pytorch dataloader
def batches(x: pd.DataFrame, y: pd.DataFrame, batch_size: int, shuffle: bool = True):
    if shuffle:
        idx = np.random.permutation(x.shape[0])
    else:
        idx = np.arange(x.shape[0])
    for i in range(0, len(idx), batch_size):
        idx_ = idx[i : i + batch_size]
        batch_y = y.iloc[idx_]
        batch_x = x.iloc[idx_]
        logging.debug(f"Batch {i // batch_size + 1}/{len(idx) // batch_size + 1} size: {len(idx_)}")
        yield batch_x, batch_y


def get_model_class():
    config = load_config()
    model_module = importlib.import_module(config["model"]["code"]["module"])
    model_class = getattr(model_module, config["model"]["code"]["class"])
    importlib.import_module(config["model"]["code"]["module"])
    return model_class


def dummy_inference(model: onnx.ModelProto):
    onnx.checker.check_model(model, full_check=True)
    inputs = {}
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    for node in sess.get_inputs():
        input_ = np.array([[None]])
        if node.type == "tensor(string)":
            input_ = input_.astype(np.str_)
        else:
            input_ = input_.astype(np.float32)
        inputs[node.name] = input_
    outputs = [node.name for node in sess.get_outputs()]
    dat = sess.run(outputs, inputs)
    dat = dict(zip(outputs, dat))
    return dat
