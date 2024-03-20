from onnxconverter_common import FloatTensorType, StringTensorType, TensorType
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType
from juniper.preprocessor.column_normalizer import ColumnNormalizer


def feature_type_to_onnx_type(feature_type: FeatureType) -> TensorType:
    match feature_type:
        case FeatureType.ARRAY:
            raise ValueError("Must not call this function with array type")
        case FeatureType.NUMERIC:
            return FloatTensorType([None, 1])
        case FeatureType.CATEGORICAL:
            return StringTensorType([None, 1])
        case FeatureType.BOOLEAN:
            return FloatTensorType([None, 1])
        case FeatureType.TIMESTAMP:
            raise NotImplementedError("Timestamps not yet supported")
        case _:
            raise ValueError(f"Unknown feature type {feature_type}")


def get_onnx_types(column_transformer) -> list[tuple[str, TensorType]]:
    initial_types = []
    for name, t, columns in column_transformer.transformers:
        if isinstance(t, ColumnNormalizer):
            raise NotImplementedError("Array types not yet supported")
        for col in columns:
            initial_types.append((col, feature_type_to_onnx_type(name)))
    return initial_types


def to_onnx(column_transformer: ColumnTransformer, name: str):
    model_onnx = convert_sklearn(
        model=column_transformer,
        name=name,
        initial_types=get_onnx_types(column_transformer),
    )
    return model_onnx
