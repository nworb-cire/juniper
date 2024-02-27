import enum


class FeatureType(enum.StrEnum):
    UNUSABLE = "unusable"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    STRING = "string"
    ARRAY = "array"
