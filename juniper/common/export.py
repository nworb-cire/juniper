from onnxconverter_common import FloatTensorType, StringTensorType, TensorType
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType


def get_onnx_types(column_transformer) -> list[tuple[str, TensorType]]:
    initial_types = []
    for name, _, columns in column_transformer.transformers:
        for col in columns:
            match name:
                case FeatureType.NUMERIC:
                    initial_types.append((col, FloatTensorType([None, 1])))
                case FeatureType.CATEGORICAL:
                    initial_types.append((col, StringTensorType([None, 1])))
                case FeatureType.BOOLEAN:
                    initial_types.append((col, FloatTensorType([None, 1])))
                case FeatureType.TIMESTAMP:
                    raise NotImplementedError("Timestamps not yet supported")
                case _:
                    raise ValueError(f"Unknown feature type {name}")
    return initial_types


def to_onnx(column_transformer: ColumnTransformer, name: str):
    model_onnx = convert_sklearn(
        model=column_transformer,
        name=name,
        initial_types=get_onnx_types(column_transformer),
    )
    return model_onnx
