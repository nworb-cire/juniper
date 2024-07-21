from collections import defaultdict

import pyarrow as pa

from juniper.common.data_type import FeatureType


def get_feature_types(
    schema: pa.Schema,
    enabled_feature_types: list[FeatureType] | None = None,
    override_unusable_features: tuple[str] = (),
) -> dict[FeatureType, list[str]]:
    if enabled_feature_types is None:
        enabled_feature_types = list(FeatureType)

    columns = defaultdict(list)

    for i in range(len(schema)):
        field = schema.field(i)
        if field.name.startswith(override_unusable_features):  # TODO: glob
            columns[FeatureType.UNUSABLE].append(field.name)
        elif isinstance(field.type, pa.lib.ListType):
            if FeatureType.ARRAY in enabled_feature_types:
                columns[FeatureType.ARRAY].append(field.name)

    # Some nested array fields may appear in the schema
    for base_field_name in sorted(columns[FeatureType.ARRAY], key=len):
        for field_name in columns[FeatureType.ARRAY]:
            if field_name.startswith(base_field_name) and field_name != base_field_name:
                columns[FeatureType.UNUSABLE].append(field_name)
    columns[FeatureType.ARRAY] = [c for c in columns[FeatureType.ARRAY] if c not in columns[FeatureType.UNUSABLE]]

    for i in range(len(schema)):
        field = schema.field(i)
        if field.name in columns[FeatureType.UNUSABLE] + columns[FeatureType.ARRAY]:
            continue
        if field.name.startswith(tuple(columns[FeatureType.ARRAY])):
            # Sometimes array fields may get extracted into a flattened schema if the array has length 1
            if field.name not in columns[FeatureType.ARRAY]:
                columns[FeatureType.UNUSABLE].append(field.name)
            continue
        match field.metadata[b"usable_type"].decode():
            case FeatureType.NUMERIC:
                if FeatureType.BOOLEAN in enabled_feature_types and field.type == pa.bool_():
                    columns[FeatureType.BOOLEAN].append(field.name)
                else:
                    if FeatureType.NUMERIC in enabled_feature_types:
                        columns[FeatureType.NUMERIC].append(field.name)
            case FeatureType.CATEGORICAL:
                if FeatureType.CATEGORICAL in enabled_feature_types:
                    columns[FeatureType.CATEGORICAL].append(field.name)
            case FeatureType.BOOLEAN:
                if FeatureType.BOOLEAN in enabled_feature_types:
                    columns[FeatureType.BOOLEAN].append(field.name)
            case FeatureType.TIMESTAMP:
                if FeatureType.TIMESTAMP in enabled_feature_types:
                    columns[FeatureType.TIMESTAMP].append(field.name)
            case _:
                columns[FeatureType.UNUSABLE].append(field.name)

    for type_ in FeatureType:
        columns[type_] = list(sorted(columns[type_]))
        if not columns[type_]:
            del columns[type_]

    return columns
