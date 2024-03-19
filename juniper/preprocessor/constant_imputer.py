from typing import Any

from sklearn.impute import SimpleImputer


def ConstantImputer(fill_value: Any = None, add_indicator: bool = False, missing_values: Any = float("nan")):
    return SimpleImputer(
        strategy="constant", fill_value=fill_value, add_indicator=add_indicator, missing_values=missing_values
    )


# TODO: handle ONNX export with indicator flag
