import pandas as pd
import pyarrow as pa
from dask_ml.preprocessing import Categorizer, OrdinalEncoder
from sklearn.pipeline import Pipeline

from juniper.preprocessor.dask.cast_transformer import CastTransformer
from juniper.preprocessor.dask.column_transformer import ColumnTransformer
from juniper.preprocessor.dask.constant_imputer import ConstantImputer
from juniper.preprocessor.dask.robust_scaler import RobustScaler


def get_default_numeric_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(add_indicator=True)),
            (
                "scaler",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("typecast", CastTransformer()),
                                    ("scaler", RobustScaler(quantile_range=(1.0, 99.0))),
                                ]
                            ),
                            columns,
                        )  # Scale only the numeric columns
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
            ("categorizer", Categorizer(columns=pd.Index(columns))),
            ("encoder", OrdinalEncoder()),
            ("typecast", CastTransformer()),
        ]
    )


def get_default_boolean_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "typecast",
                Pipeline(
                    steps=[
                        ("bool", CastTransformer(dtype="boolean")),
                        ("float", CastTransformer()),
                    ]
                ),
            ),
            ("imputer", ConstantImputer(fill_value=-1)),
        ]
    )


def get_default_timestamp_pipeline(columns: list[str]) -> Pipeline:
    unix_epoch = pa.scalar(0, type=pa.timestamp("ns", tz="UTC"))
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(fill_value=unix_epoch, missing_values=pd.NA, add_indicator=True)),
            ("typecast", CastTransformer()),
            ("scaler", RobustScaler()),
        ]
    )
