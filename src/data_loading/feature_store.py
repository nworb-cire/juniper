from datetime import datetime

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from s3path import S3Path

from src.common.data_type import compute_maybe
from src.common.setup import load_config
from src.data_loading.data_source import BaseDataSource


class FeatureStore(BaseDataSource):
    def __init__(self, path: S3Path = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["feature_store"]["location"]
        self.timestamp_column = config["data_sources"]["feature_store"]["timestamp_column"]
        super().__init__(path=path)

    def get_metadata(self):
        try:
            path = next(self.path.iterdir())
        except StopIteration:
            path = self.path
        config = load_config()
        schema = pq.read_schema(
            path.as_posix()[1:],
            filesystem=pa.fs.S3FileSystem(
                endpoint_override=config["minio"]["endpoint_url"],
                access_key=config["minio"]["aws_access_key_id"],
                secret_key=config["minio"]["aws_secret_access_key"],
            ),
        )
        self.schema = schema

    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        df = self.read_parquet()
        train_idx = compute_maybe(train_idx)
        train = df.loc[train_idx]
        if test_idx is not None:
            test_idx = compute_maybe(test_idx)
            test = df.loc[test_idx]
        else:
            test = None
        return train, test
