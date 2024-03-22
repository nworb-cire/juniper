import importlib

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


def unpack(vals):
    vals = [np.vstack(v) for v in vals]
    N = max([v.shape for v in vals])
    vals = [np.pad(v, ((0, N[0] - v.shape[0]), (0, N[1] - v.shape[1])), constant_values=np.nan) for v in vals]
    vals = np.array(vals)
    return vals


def onnx_name_to_pd_name(name: str) -> str | None:
    if name == "output":
        return None
    assert name.endswith("_arr")
    name = name[:-4]
    return f"{name}__{name}"


def _to_tensor(model, x: pd.DataFrame) -> dict[str, torch.Tensor]:
    pd_names = [onnx_name_to_pd_name(name) for name in model.inputs.keys() if name != "features"]
    x_arr = {name: unpack(x[onnx_name_to_pd_name(name)].values) for name in model.inputs.keys() if name != "features"}
    for name, arr in x_arr.items():
        assert arr.shape[0] == x.shape[0]
        assert arr.shape[1] == model.inputs[name]

    x = x[[c for c in x.columns if c not in pd_names]].values
    assert x.shape[1] == model.inputs["features"]
    x = torch.tensor(x, dtype=torch.float32)

    x_arr = {k.replace(".", "_"): torch.tensor(v, dtype=torch.float32) for k, v in x_arr.items()}
    return {"features": x, **x_arr}


def batches(x: pd.DataFrame, y: pd.DataFrame, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        na_idx = y.iloc[i : i + batch_size].isna().all(axis=1)
        if na_idx.all():
            continue
        batch_y = y.iloc[i : i + batch_size].loc[~na_idx]
        batch_x = x.iloc[i : i + batch_size].loc[~na_idx]
        yield batch_x, batch_y


def get_model_class():
    config = load_config()
    model_module = importlib.import_module(config["model"]["module"])
    model_class = getattr(model_module, config["model"]["class"])
    importlib.import_module(config["model"]["module"])
    return model_class
