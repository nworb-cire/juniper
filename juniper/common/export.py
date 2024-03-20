from onnxconverter_common import FloatTensorType, StringTensorType, TensorType
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType
from juniper.preprocessor.column_normalizer import ColumnNormalizer


def feature_type_to_onnx_type(feature_type: FeatureType, arr: bool = False) -> TensorType:
    dim = 1 if not arr else None
    match feature_type:
        case FeatureType.ARRAY:
            raise ValueError("Must not call this function with array type")
        case FeatureType.NUMERIC:
            return FloatTensorType([dim, 1])
        case FeatureType.CATEGORICAL:
            return StringTensorType([dim, 1])
        case FeatureType.BOOLEAN:
            return FloatTensorType([dim, 1])
        case FeatureType.TIMESTAMP:
            raise NotImplementedError("Timestamps not yet supported")
        case _:
            raise ValueError(f"Unknown feature type {feature_type}")


def get_onnx_types(column_transformer: ColumnTransformer) -> list[tuple[str, TensorType]]:
    initial_types = []
    for name, _, columns in column_transformer.transformers:
        if name == "remainder":
            continue
        for col in columns:
            if "." in name:
                col_type = name.split(".")[-1]
                arr = True
            else:
                col_type = name
                arr = False
            initial_types.append((col, feature_type_to_onnx_type(col_type, arr)))
    return initial_types


def to_onnx(column_transformer: ColumnTransformer, name: str):
    transformers = []
    sub_transformers = []
    for name_, t, cols in column_transformer.transformers_:
        if isinstance(t, ColumnNormalizer):
            sub_transformers.append(to_onnx(t.column_transformer, f"{name}_{name_}"))
        else:
            transformers.append((name_, t, cols))
    ct_out = ColumnTransformer(transformers, remainder="drop")
    ct_out.transformers_ = transformers

    model_onnx = convert_sklearn(
        model=ct_out,
        name=name,
        initial_types=get_onnx_types(ct_out),
        naming=name + "_",
    )
    for sub in sub_transformers:
        # doing model_onnx.MergeFrom(sub) does not work due to skl2onnx potentially using different opset versions
        # for the sub-models
        sub.MergeFrom(model_onnx)
        model_onnx = sub
    return model_onnx
