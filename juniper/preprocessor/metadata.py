import pyarrow as pa

from juniper.common.setup import load_config


class FeatureStoreMetadata:
    # TODO: would be nice not to have to redefine these
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    timestamp_columns: list[str]
    unusable_columns: list[str]

    def __init__(self, schema: pa.Schema, separate_booleans: bool = True):
        self.schema = schema

        config = load_config()
        enabled_feature_types = config["data_sources"]["feature_store"]["enabled_feature_types"]

        self.numeric_columns = []
        self.categorical_columns = []
        self.boolean_columns = []
        self.timestamp_columns = []
        self.array_columns = []
        self.unusable_columns = []

        for field_name in self.schema.names:
            field = self.schema.field(field_name)
            if isinstance(field.type, pa.lib.ListType):
                if "array" in enabled_feature_types:
                    self.array_columns.append(field_name)
                    continue
            match field.metadata[b"usable_type"].decode():
                case "numeric":
                    if separate_booleans and field.type == pa.bool_():
                        if "boolean" in enabled_feature_types:
                            self.boolean_columns.append(field_name)
                    else:
                        if "numeric" in enabled_feature_types:
                            self.numeric_columns.append(field_name)
                case "categorical":
                    if "categorical" in enabled_feature_types:
                        self.categorical_columns.append(field_name)
                case "boolean":
                    if "boolean" in enabled_feature_types:
                        self.boolean_columns.append(field_name)
                case "timestamp":
                    if "timestamp" in enabled_feature_types:
                        self.timestamp_columns.append(field_name)
                case _:
                    self.unusable_columns.append(field_name)
