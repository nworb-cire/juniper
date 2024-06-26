import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

from juniper.preprocessor.cast_transformer import CastTransformer, DatetimeCastTransformer
from juniper.preprocessor.constant_imputer import ConstantImputer


def get_default_numeric_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("typecast", CastTransformer()),
            ("imputer", ConstantImputer(add_indicator=False)),  # FIXME: indicator not exporting
            ("scaler", RobustScaler(quantile_range=(1.0, 99.0))),
            ("typecast2", CastTransformer()),
        ]
    )


def get_default_categorical_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("imputer", ConstantImputer(fill_value=-1)),
            ("typecast", CastTransformer()),
        ]
    )


def get_default_boolean_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("typecast", CastTransformer()),
            ("imputer", ConstantImputer(fill_value=-1)),
        ]
    )


def get_default_timestamp_pipeline(columns: list[str]) -> Pipeline:
    unix_epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(fill_value=unix_epoch, missing_values=pd.NA, add_indicator=True)),
            (
                "scaler",
                ColumnTransformer(
                    transformers=[
                        (
                            "timestamp",
                            Pipeline(
                                steps=[
                                    ("typecast", DatetimeCastTransformer()),
                                    ("scaler", RobustScaler()),
                                ]
                            ),
                            columns,
                        )
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            ("typecast", CastTransformer()),
        ]
    )
