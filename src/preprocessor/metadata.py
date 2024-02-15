import pyarrow as pa


class FeatureStoreMetadata:
    # TODO: would be nice not to have to redefine these
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    timestamp_columns: list[str]
    unusable_columns: list[str]
    
    def __init__(self, schema: pa.Schema, recurse_arrays: bool = False, separate_booleans : bool = True):
        self.schema = schema

        self.numeric_columns = []
        self.categorical_columns = []
        self.boolean_columns = []
        self.timestamp_columns = []
        self.unusable_columns = []

        for field_name in self.schema.names:
            field = self.schema.field(field_name)
            if isinstance(field.type, pa.lib.ListType):
                if recurse_arrays:
                    pass
                else:
                    self.unusable_columns.append(field_name)
                    continue
            match field.metadata[b'usable_type'].decode():
                case 'numeric':
                    if separate_booleans and field.type == pa.bool_():
                        self.boolean_columns.append(field_name)
                    else:
                        self.numeric_columns.append(field_name)
                case 'categorical':
                    self.categorical_columns.append(field_name)
                case 'boolean':
                    self.boolean_columns.append(field_name)
                case 'timestamp':
                    self.timestamp_columns.append(field_name)
                case _:
                    self.unusable_columns.append(field_name)
