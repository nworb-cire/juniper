import json

import pyarrow as pa


def _get_flattened_fields(list_type: pa.lib.ListType, record_path: str, metadata: dict) \
        -> list[tuple[str, pa.lib.DataType]]:
    fields = []
    for field in list_type.value_field.flatten():
        name = field.name.replace("item", record_path)
        _metadata = metadata.get(field.name.replace("item.", ""))
        if isinstance(field.type, pa.lib.ListType):
            fields.extend(_get_flattened_fields(field.type, name, _metadata[0]))
        else:
            assert isinstance(_metadata, str), f"Unknown metadata type: {_metadata}"
            fields.append(pa.field(name, field.type, metadata={"usable_type": _metadata}))
    return fields

def get_field_schema(field: pa.lib.Field) -> pa.Schema:
    if not isinstance(field.type, pa.lib.ListType):
        raise ValueError("Field type must be a ListType")
    fields = _get_flattened_fields(
        field.type, field.name, 
        metadata=json.loads(field.metadata[b'usable_type'].decode("utf-8"))[0])
    return pa.schema(fields)
