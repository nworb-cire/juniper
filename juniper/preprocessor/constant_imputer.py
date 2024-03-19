from typing import Any

from skl2onnx import update_registered_converter
from skl2onnx.operator_converters.imputer_op import convert_sklearn_imputer
from skl2onnx.shape_calculators.imputer import calculate_sklearn_imputer_output_shapes
from sklearn.impute import SimpleImputer


class ConstantImputer(SimpleImputer):
    def __init__(self, fill_value: Any = None, add_indicator: bool = False, missing_values: Any = float("nan")):
        super().__init__(
            strategy="constant", fill_value=fill_value, add_indicator=add_indicator, missing_values=missing_values
        )


update_registered_converter(
    ConstantImputer,
    "JuniperConstantImputer",
    calculate_sklearn_imputer_output_shapes,
    convert_sklearn_imputer,
)
