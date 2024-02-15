import pandas as pd
import pyarrow as pa
from dask_ml.preprocessing import Categorizer, OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.preprocessor.dask.cast_transformer import CastTransformer
from src.preprocessor.dask.constant_imputer import ConstantImputer
from src.preprocessor.dask.robust_scaler import RobustScaler


def get_default_numeric_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", ConstantImputer(add_indicator=True)),
            ("scaler", RobustScaler(quantile_range=(1.0, 99.0))),
            ("typecast", CastTransformer()),
        ]
    )

def get_default_categorical_pipeline(columns: list[str]) -> Pipeline:
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
            ("typecast", CastTransformer()),
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
