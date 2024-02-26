import enum
from typing import TypeVar


class FeatureType(enum.StrEnum):
    UNUSABLE = "unusable"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    STRING = "string"
    ARRAY = "array"


T = TypeVar("T")


def compute_maybe(df: T) -> T:
    if hasattr(df, "compute"):
        return df.compute()
    return df
