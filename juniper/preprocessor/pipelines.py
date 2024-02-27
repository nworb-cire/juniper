import numpy as np
import pandas as pd
from skl2onnx.sklapi import CastTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

from juniper.preprocessor.constant_imputer import ConstantImputer


def get_default_numeric_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(add_indicator=True)),
            (
                "scaler",
                ColumnTransformer(  # Scale only the numeric columns
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("typecast", CastTransformer()),
                                    ("scaler", RobustScaler(quantile_range=(1.0, 99.0))),
                                ]
                            ),
                            slice(len(columns)),  # This needs to be a slice since the imputer returns a np array
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("typecast", CastTransformer()),
        ]
    )


def get_default_categorical_pipeline(columns: list[str]) -> Pipeline:
    # TODO: Handle "not before seen" categories in deployment
    return Pipeline(
        steps=[
            # ("categorizer", Categorizer(columns=pd.Index(columns))),
            ("imputer", ConstantImputer(fill_value="_missing")),
            ("encoder", OrdinalEncoder()),
            ("typecast", CastTransformer()),
        ]
    )


def get_default_boolean_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("nan_imputer", ConstantImputer(fill_value=np.nan)),
            ("float", CastTransformer()),
            ("imputer", ConstantImputer(fill_value=-1)),
        ]
    )


def get_default_timestamp_pipeline(columns: list[str]) -> Pipeline:
    unix_epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(fill_value=unix_epoch, missing_values=pd.NA, add_indicator=True)),
            (
                "typecast",
                Pipeline(
                    steps=[
                        ("np_datetime", CastTransformer(dtype=np.datetime64)),
                        ("np_int", CastTransformer(dtype=np.int64)),
                        ("float", CastTransformer()),
                    ]
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )
