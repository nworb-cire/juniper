from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from pyarrow import dataset as ds
from pyarrow import compute as pc

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.data_loading.data_source import (
    BaseDataSource,
    S3ParquetDataSource,
    LocalDataSource,
    ParquetDataSource,
)


class BaseFeatureStore(BaseDataSource, ABC):
    config_location = "feature_store"

    def get_metadata(self):
        self.schema = self.get_schema()
        self.feature_types = self.get_feature_types(self.schema)
        return self.feature_types

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        pass

    @classmethod
    @abstractmethod
    def get_feature_types(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
        pass


class BaseParquetFeatureStore(BaseFeatureStore, ParquetDataSource, ABC):
    def __init__(self, path: Path | None = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["feature_store"]["location"]
        self.path = path
        super().__init__()

    def _load_train_test(
        self,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        train_time_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        filters = ~ds.field(self.index_column).is_null()
        if train_idx is not None:
            filters &= pc.is_in(ds.field(self.index_column), pa.array(train_idx))
        train = self.read_parquet(filters=filters)
        train = train.sort_values(self.timestamp_column)
        train = train[~train.index.duplicated(keep="last")]
        if test_idx is not None:
            filters = ~ds.field(self.index_column).is_null() & pc.is_in(ds.field(self.index_column), pa.array(test_idx))
            test = self.read_parquet(filters=filters)
            test = test.sort_values(self.timestamp_column)
            test = test[~test.index.duplicated(keep="last")]
        else:
            test = None
        return train, test


class ParquetFeatureStore(BaseParquetFeatureStore, ABC):
    @classmethod
    def get_feature_types(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
        columns = defaultdict(list)

        config = load_config()
        enabled_feature_types = config["data_sources"]["feature_store"]["enabled_feature_types"]
        override_unusable_features = tuple(config["data_sources"]["feature_store"].get("unusable_features", []))

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


class LocalParquetFeatureStore(ParquetFeatureStore, LocalDataSource):
    def get_schema(self) -> pa.Schema:
        if self.path.is_dir():
            path = next(self.path.iterdir())
        else:
            path = self.path
        return pq.read_schema(path)


class S3ParquetParquetFeatureStore(ParquetFeatureStore, S3ParquetDataSource):
    def get_schema(self) -> pa.Schema:
        try:
            path = next(self.path.iterdir())
        except StopIteration:
            path = self.path
        config = load_config()
        return pq.read_schema(
            path.as_posix()[1:],
            filesystem=pa.fs.S3FileSystem(
                endpoint_override=config["minio"]["endpoint_url"],
                access_key=config["minio"]["aws_access_key_id"],
                secret_key=config["minio"]["aws_secret_access_key"],
            ),
        )


# class SqlFeatureStore(BaseFeatureStore, SqlDataSource):
#     def __init__(self, connection_str: str | None = None):
#         config = load_config()
#         if connection_str is None:
#             connection_str = config["data_sources"]["feature_store"]["location"]
#         self.connection_str = connection_str
#         query = config["data_sources"]["feature_store"]["query"]
#         self.query = query
#         super().__init__()
#
#     def get_schema(self) -> pa.Schema:
#         pass
#
#     @classmethod
#     def get_feature_types(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
#         raise NotImplementedError
#
#     def _load_train_test(
#         self,
#         train_idx: pd.Index | None = None,
#         test_idx: pd.Index | None = None,
#         train_time_end: pd.Timestamp | None = None,
#     ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
#         idx = train_idx.tolist() + (test_idx.tolist() if test_idx is not None else [])
#         df = self.read_sql(self.query.format(idx=tuple(idx)))
#         train = df[df[self.index_column].isin(train_idx)]
#         if test_idx is not None:
#             test = df[df[self.index_column].isin(test_idx)]
#         else:
#             test = None
#         return train, test
