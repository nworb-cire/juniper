import json
from collections import defaultdict
from datetime import datetime

import onnx
from onnxconverter_common import FloatTensorType, StringTensorType, TensorType
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.preprocessor.column_normalizer import ColumnNormalizer


def get_common_opset(*models: list[onnx.ModelProto]):
    ret = defaultdict(int)
    for model in models:
        for opset in model.opset_import:
            ret[opset.domain] = max(ret[opset.domain], opset.version)
    return [{"domain": k, "version": v} for k, v in ret.items()]


def clear_opset(model: onnx.ModelProto):
    while len(model.opset_import) > 0:
        model.opset_import.pop()


def set_opset(model: onnx.ModelProto, version: int, domain: str = None):
    opset = model.opset_import.add()
    opset.version = version
    if domain is not None:
        opset.domain = domain


def merge_models(m1: onnx.ModelProto, m2: onnx.ModelProto, io_mapping: list[tuple[str, str]]) -> onnx.ModelProto:
    common_opset = get_common_opset(m1, m2)
    clear_opset(m1)
    clear_opset(m2)
    for opset in common_opset:
        set_opset(m1, opset["version"], opset["domain"])
        set_opset(m2, opset["version"], opset["domain"])
    return onnx.compose.merge_models(m1, m2, io_mapping)


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


def _to_onnx(column_transformer: ColumnTransformer, name: str | None = None):
    if name is not None:
        name = name.replace(".", "_")
    transformers = []
    sub_transformers = []
    for name_, t, cols in column_transformer.transformers_:
        if isinstance(t, ColumnNormalizer):
            sub_transformers.append(_to_onnx(t.column_transformer, name_))
        else:
            transformers.append((name_, t, cols))
    ct_out = ColumnTransformer(transformers, remainder="drop")
    ct_out.transformers_ = transformers

    model_onnx = convert_sklearn(
        model=ct_out,
        name=name if name is not None else "base",
        initial_types=get_onnx_types(ct_out),
        naming=name + "_" if name is not None else "",
        target_opset=18,
    )
    # rename output
    assert len(model_onnx.graph.output) == 1
    output_node_name = model_onnx.graph.output[0].name
    renamed_node_name = "features" if name is None else f"{name}_arr"
    model_onnx.graph.output[0].name = renamed_node_name
    for node in model_onnx.graph.node:
        for i in range(len(node.output)):
            if node.output[i] == output_node_name:
                node.output[i] = renamed_node_name
    # merge subgraphs
    for sub in sub_transformers:
        model_onnx = merge_models(model_onnx, sub, [])
    return model_onnx


def add_metadata(model_onnx: onnx.ModelProto, key: str, value: str):
    message_proto = onnx.StringStringEntryProto()
    message_proto.key = key
    message_proto.value = value
    model_onnx.metadata_props.append(message_proto)


def add_default_metadata(model_onnx: onnx.ModelProto):
    config = load_config()
    enabled_feature_types = config["data_sources"]["feature_store"]["enabled_feature_types"]
    if FeatureType.ARRAY in enabled_feature_types:
        feature_meta = config["data_sources"]["feature_store"].get("feature_meta", {})
        add_metadata(model_onnx, "feature_meta", json.dumps(feature_meta))
    add_metadata(model_onnx, "creation_date", str(datetime.utcnow()))
    for k, v in config.get("model_info", {}).items():
        add_metadata(model_onnx, k, v)


def to_onnx(column_transformer: ColumnTransformer):
    model_onnx = _to_onnx(column_transformer)
    add_default_metadata(model_onnx)
    return model_onnx
