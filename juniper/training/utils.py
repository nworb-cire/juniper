import importlib
import logging

import numpy as np
import pandas as pd
import torch

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


def unpack(vals: list[np.ndarray]):
    vals = [np.vstack(v) for v in vals]
    N = max([v.shape for v in vals])
    vals = [np.pad(v, ((0, N[0] - v.shape[0]), (0, N[1] - v.shape[1])), constant_values=np.nan) for v in vals]
    vals = np.array(vals)
    return vals


def onnx_name_to_pd_name(name: str) -> str | None:
    assert name != "features"
    name = name.replace(".", "_")
    return f"{name}__{name}"


def _to_tensor(model, x: pd.DataFrame) -> dict[str, torch.Tensor]:
    x_arr = {name: unpack(x[onnx_name_to_pd_name(name)].values) for name in model.inputs.keys() if name != "features"}
    for name, arr in x_arr.items():
        assert arr.shape[0] == x.shape[0]
        assert arr.shape[1] == len(model.inputs[name])

    x = x.drop(columns=[onnx_name_to_pd_name(name) for name in model.inputs.keys() if name != "features"])
    assert x.shape[1] == len(model.inputs["features"])
    x = torch.tensor(x, dtype=torch.float32)

    x_arr = {k.replace(".", "_"): torch.tensor(v, dtype=torch.float32) for k, v in x_arr.items()}
    return {"features": x, **x_arr}


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
    model_module = importlib.import_module(config["model"]["module"])
    model_class = getattr(model_module, config["model"]["class"])
    importlib.import_module(config["model"]["module"])
    return model_class
