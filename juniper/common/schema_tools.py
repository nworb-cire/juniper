import json
from collections import defaultdict

import pyarrow as pa
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType


def _get_flattened_fields(
    list_type: pa.lib.ListType, record_path: str, metadata: dict | None
) -> list[tuple[str, pa.lib.DataType]]:
    fields = []
    for field in list_type.value_field.flatten():
        name = field.name.replace("element", record_path)
        if isinstance(metadata, str):
            _metadata = metadata
        elif metadata is not None:
            _metadata = metadata.get(field.name.replace("element.", ""))
        else:
            _metadata = None
        if isinstance(field.type, pa.lib.ListType):
            if _metadata is None:
                _metadata = [None]
            fields.extend(_get_flattened_fields(field.type, name, _metadata[0]))
        else:
            if _metadata is None:
                _metadata = "unusable"
            assert isinstance(_metadata, str), f"Unknown metadata type: {_metadata}"
            fields.append(pa.field(name, field.type, metadata={"usable_type": _metadata}))
    return fields


def get_field_schema(field: pa.lib.Field) -> pa.Schema:
    if not isinstance(field.type, pa.lib.ListType):
        raise ValueError("Field type must be a ListType")
    if (metadata_str := field.metadata.get(b"usable_type", b"").decode()) != FeatureType.UNUSABLE:
        metadata = json.loads(metadata_str)[0]
        if metadata == FeatureType.UNUSABLE:
            metadata = None
    else:
        metadata = None
    fields = _get_flattened_fields(field.type, field.name, metadata=metadata)
    return pa.schema(fields)


def get_input_mapping(preprocessor: ColumnTransformer) -> dict[str, list[str]]:
    """
    :param preprocessor: Fitted ColumnTransformer
    :return: Inverse mapping of preprocessor outputs to preprocessor inputs
    """
    ret: dict[str, list[str]] = defaultdict(list)
    for name, slice_ in preprocessor.output_indices_.items():
        if name == "remainder":
            continue
        elif name in [FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.BOOLEAN, FeatureType.TIMESTAMP]:
            ret["features"] += list(preprocessor.feature_names_in_[slice_])
        else:  # Array features
            assert name in preprocessor.named_transformers_
            # assert isinstance(preprocessor.named_transformers_[name], ColumnNormalizer)
            ret[name] = list(preprocessor.named_transformers_[name].schema_out.names)
    return dict(ret)
