from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from s3path import S3Path

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.data_loading.data_source import S3DataSource


class BaseFeatureStore(S3DataSource, ABC):
    def __init__(self, path: S3Path = None):
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
        df = self.read_parquet(
            filters=[(self.index_column, "in", train_idx.union(test_idx).tolist())],
        )
        train = df.reindex(train_idx)
        if test_idx is not None:
            test = df.reindex(test_idx)
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


class ParquetFeatureStore(BaseFeatureStore):
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

    @classmethod
    def get_feature_metadata(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
        columns = defaultdict(list)

        enabled_feature_types = load_config()["data_sources"]["feature_store"]["enabled_feature_types"]

        for i in range(len(schema)):
            field = schema.field(i)
            if isinstance(field.type, pa.lib.ListType):
                if FeatureType.ARRAY in enabled_feature_types:
                    columns[FeatureType.ARRAY].append(field.name)
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

        return columns
