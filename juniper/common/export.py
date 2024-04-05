import dataclasses
import json
from datetime import datetime
from functools import reduce, partial

import onnx
from onnxconverter_common import FloatTensorType, StringTensorType, TensorType
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.training.metrics import EvalMetrics


def set_opset(model: onnx.ModelProto, opset: list[onnx.OperatorSetIdProto]) -> onnx.ModelProto:
    while len(model.opset_import) > 0:
        model.opset_import.pop()
    model.opset_import.extend(opset)
    return model


def merge_models(m1: onnx.ModelProto, m2: onnx.ModelProto, io_map: list[tuple[str, str]]) -> onnx.ModelProto:
    opset = [
        onnx.helper.make_opsetid("", 17),
        onnx.helper.make_opsetid("ai.onnx.ml", 2),
    ]
    set_opset(m1, opset)
    set_opset(m2, opset)
    new_model = onnx.compose.merge_models(
        m1=m1,
        m2=m2,
        io_map=io_map,
        producer_name="juniper",
    )
    set_opset(new_model, opset)
    onnx.checker.check_model(new_model, full_check=True)
    return new_model


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
    transformers = []
    sub_transformers = []
    for name_, t, cols in column_transformer.transformers_:
        if isinstance(t, ColumnNormalizer):
            sub_transformers.append(_to_onnx(t.column_transformer, name_))
        elif name_ == "remainder":
            continue
        else:
            transformers.append((name_, t, cols))
    ct_out = ColumnTransformer(transformers, remainder="drop")
    ct_out.transformers_ = transformers

    model_onnx = convert_sklearn(
        model=ct_out,
        name=name if name is not None else "base",
        initial_types=get_onnx_types(ct_out),
        naming=name + "_" if name is not None else "",
        target_opset=17,
    )
    onnx.checker.check_model(model_onnx, full_check=True)
    # rename output
    assert len(model_onnx.graph.output) == 1
    output_node_name = model_onnx.graph.output[0].name
    renamed_node_name = "features" if name is None else name
    model_onnx.graph.output[0].name = renamed_node_name
    for node in model_onnx.graph.node:
        for i in range(len(node.output)):
            if node.output[i] == output_node_name:
                node.output[i] = renamed_node_name
    # merge subgraphs
    model_onnx = reduce(partial(merge_models, io_map=[]), sub_transformers, model_onnx)
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
    for k, v in config["model"].get("metadata", {}).items():
        if k == "doc_string":
            model_onnx.doc_string = v
        else:
            add_metadata(model_onnx, k, v)


def to_onnx(column_transformer: ColumnTransformer):
    model_onnx = _to_onnx(column_transformer)
    add_default_metadata(model_onnx)
    onnx.checker.check_model(model_onnx, full_check=True)
    return model_onnx


def add_metrics(model_onnx: onnx.ModelProto, metrics: list[EvalMetrics]):
    data = json.dumps([dataclasses.asdict(m) for m in metrics])
    message_proto = onnx.StringStringEntryProto()
    message_proto.key = "metrics"
    message_proto.value = data
    model_onnx.metadata_props.append(message_proto)
