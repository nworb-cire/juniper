from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.data_loading.data_source import BaseDataSource, S3DataSource


class BaseFeatureStore(BaseDataSource, ABC):
    def __init__(self, path: Path = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["feature_store"]["location"]
        self.timestamp_column = config["data_sources"]["feature_store"]["timestamp_column"]
        super().__init__(path=path)

    def get_metadata(self):
        self.schema = self.get_schema()
        self.metadata = self.get_feature_metadata(self.schema)

    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        train = self.read_parquet(
            filters=[(self.index_column, "in", train_idx.tolist())],
        )
        if test_idx is not None:
            test = self.read_parquet(
                filters=[(self.index_column, "in", test_idx.tolist())],
            )
        else:
            test = None
        return train, test

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        pass

    @classmethod
    @abstractmethod
    def get_feature_metadata(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
        pass


class ParquetFeatureStore(BaseFeatureStore, ABC):
    @classmethod
    def get_feature_metadata(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
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


class LocalParquetFeatureStore(ParquetFeatureStore):
    def get_schema(self) -> pa.Schema:
        if self.path.is_dir():
            path = next(self.path.iterdir())
        else:
            path = self.path
        return pq.read_schema(path)

    def read_parquet(
        self, path: Path = None, columns: list[str] = None, filters: list[tuple] | list[list[tuple]] | None = None
    ) -> pd.DataFrame:
        df = pd.read_parquet(self.path, columns=columns, filters=filters)
        return df.set_index(self.index_column)


class S3ParquetFeatureStore(ParquetFeatureStore, S3DataSource):
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
