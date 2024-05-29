import dataclasses
import json
from collections import defaultdict
from datetime import datetime

import onnx
import sklearn.compose
from onnxconverter_common import FloatTensorType, StringTensorType, TensorType

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.modeling.metrics import EvalMetrics


def get_common_opset(m1: onnx.ModelProto, m2: onnx.ModelProto) -> list[onnx.OperatorSetIdProto]:
    opset: dict[str, int] = defaultdict(int)
    for m in (m1, m2):
        for op in m.opset_import:
            opset[op.domain] = max(opset[op.domain], op.version)
    return [onnx.helper.make_opsetid(domain, version) for domain, version in opset.items()]


def set_opset(model: onnx.ModelProto, opset: list[onnx.OperatorSetIdProto]) -> onnx.ModelProto:
    while len(model.opset_import) > 0:
        model.opset_import.pop()
    model.opset_import.extend(opset)
    return model


def merge_models(m1: onnx.ModelProto, m2: onnx.ModelProto, io_map: list[tuple[str, str]]) -> onnx.ModelProto:
    opset = get_common_opset(m1, m2)
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
            # FIXME: Timestamps must be converted to unix epoch in the runner
            return FloatTensorType([dim, 1])
        case _:
            raise ValueError(f"Unknown feature type {feature_type}")


def get_onnx_types(column_transformer: sklearn.compose.ColumnTransformer) -> list[tuple[str, TensorType]]:
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
            model_onnx.graph.doc_string = v
        else:
            add_metadata(model_onnx, k, v)
    add_metadata(model_onnx, "hyperparameters", json.dumps(config["model"]["hyperparameters"]))


def add_metrics(model_onnx: onnx.ModelProto, metrics: list[EvalMetrics]):
    data = json.dumps([dataclasses.asdict(m) for m in metrics])
    message_proto = onnx.StringStringEntryProto()
    message_proto.key = "metrics"
    message_proto.value = data
    model_onnx.metadata_props.append(message_proto)
