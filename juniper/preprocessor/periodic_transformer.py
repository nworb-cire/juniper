from collections import OrderedDict

import numpy as np
import pandas as pd
from onnx import TensorProto
from onnxconverter_common import (
    Scope,
    Operator,
    ModelComponentContainer,
    FloatTensorType,
    check_input_and_output_numbers,
)
from skl2onnx import update_registered_converter

from sklearn.base import TransformerMixin, BaseEstimator


PERIODS = OrderedDict(
    {
        "year": 365.2425 * 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
        "day": 24 * 60 * 60,
    }
)


class PeriodicTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, periods: OrderedDict[str, float] = PERIODS, keep_original: bool = False):
        self.periods = periods
        self.transform_ = "pandas"
        self.keep_original = keep_original

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        out_sin = np.concatenate([np.sin(2 * np.pi * X.values / period) for period in self.periods.values()], axis=1)
        out_cos = np.concatenate([np.cos(2 * np.pi * X.values / period) for period in self.periods.values()], axis=1)
        if self.transform_ == "pandas":
            out_sin = pd.DataFrame(
                out_sin,
                columns=[f"{col}.{p}.sin" for p in self.periods.keys() for col in X.columns],
                index=X.index,
            )
            out_cos = pd.DataFrame(
                out_cos,
                columns=[f"{col}.{p}.cos" for p in self.periods.keys() for col in X.columns],
                index=X.index,
            )
            if self.keep_original:
                vals = [X]
            else:
                vals = []
            vals.extend([out_sin, out_cos])
            return pd.concat(vals, axis=1)
        if self.keep_original:
            vals = [X.values]
        else:
            vals = []
        vals.extend([out_sin, out_cos])
        return np.concatenate(vals, axis=1)

    def set_output(self, *, transform=None):
        if transform not in (None, "pandas"):
            raise ValueError(f"Invalid transform: {transform}")
        self.transform_ = transform


def calculate_juniper_periodic_transformer(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    op = operator.raw_operator
    dims = operator.inputs[0].type.shape
    if len(dims) != 2:
        raise RuntimeError("Expecting 2D input.")
    otype = FloatTensorType([dims[0], len(op.periods) * 2 * dims[1]])
    operator.outputs[0].type = otype


def convert_juniper_periodic_transformer(scope: Scope, operator: Operator, container: ModelComponentContainer):
    op = operator.raw_operator
    inp = operator.inputs[0]

    if op.keep_original:
        raise NotImplementedError

    mul = 2 * np.pi / np.array(list(op.periods.values()))
    scale_name = f"{inp.full_name}.scale"
    container.add_initializer(scale_name, TensorProto.FLOAT, [1, mul.shape[0], 1], mul.flatten())

    # (batch, n) -> (batch, 1, n)
    unsqueeze_name = f"{inp.full_name}.unsqueeze_axis"
    container.add_initializer(unsqueeze_name, TensorProto.INT64, [1], [1])
    container.add_node(
        op_type="Unsqueeze",
        inputs=[inp.full_name, unsqueeze_name],
        outputs=[f"{inp.full_name}.unsqueeze"],
        name="Unsqueeze",
        op_version=13,
    )
    # (batch, 1, n) -> (batch, p, n)
    container.add_node(
        op_type="Mul",
        inputs=[f"{inp.full_name}.unsqueeze", scale_name],
        outputs=[f"{inp.full_name}.mul"],
        name="Scale",
        op_version=14,
    )
    # (batch, p, n) -> (batch, p * n)
    reshape_name = f"{inp.full_name}.reshape_shape"
    container.add_initializer(
        reshape_name, TensorProto.INT64, [2], [inp.type.shape[0], inp.type.shape[1] * len(op.periods)]
    )
    container.add_node(
        op_type="Reshape",
        inputs=[f"{inp.full_name}.mul", reshape_name],
        outputs=[f"{inp.full_name}.reshape"],
        name="Reshape",
        op_version=14,
    )
    container.add_node(
        op_type="Sin",
        inputs=[f"{inp.full_name}.reshape"],
        outputs=[f"{inp.full_name}.sin"],
        name="Sin",
        op_version=17,
    )
    container.add_node(
        op_type="Cos",
        inputs=[f"{inp.full_name}.reshape"],
        outputs=[f"{inp.full_name}.cos"],
        name="Cos",
        op_version=17,
    )
    # 2 * (batch, n * p) -> (batch, n * p * 2)
    container.add_node(
        op_type="Concat",
        inputs=[f"{inp.full_name}.sin", f"{inp.full_name}.cos"],
        outputs=[operator.outputs[0].full_name],
        name="Concat",
        axis=1,
        op_version=13,
    )


update_registered_converter(
    PeriodicTransformer,
    "JuniperPeriodicTransformer",
    calculate_juniper_periodic_transformer,
    convert_juniper_periodic_transformer,
)
