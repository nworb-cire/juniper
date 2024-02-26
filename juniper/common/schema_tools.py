import json

import pyarrow as pa

from juniper.common.data_type import FeatureType


def _get_flattened_fields(
    list_type: pa.lib.ListType, record_path: str, metadata: dict | None
) -> list[tuple[str, pa.lib.DataType]]:
    fields = []
    for field in list_type.value_field.flatten():
        name = field.name.replace("element", record_path)
        if metadata is not None:
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
    if isinstance(metadata, str):
        return pa.schema([field.with_metadata({"usable_type": metadata})])
    fields = _get_flattened_fields(field.type, field.name, metadata=metadata)
    return pa.schema(fields)
