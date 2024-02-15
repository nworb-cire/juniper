import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
from dask import dataframe as dd
from s3path import S3Path

from src.common.setup import load_config


class BaseDataSource(ABC):
    index_column: str
    timestamp_column: str

    def __init__(
        self,
        path: S3Path,
    ):
        self.path = path
        config = load_config()
        self.index_column = config["data_sources"]["index_column"]
        self.get_metadata()

    def read_parquet(self, path: S3Path = None, columns: list[str] = None, index_column: str = None) -> pd.DataFrame:
        if path is None:
            path = self.path
        if index_column is None:
            index_column = self.index_column
        config = load_config()
        return dd.read_parquet(
            path.as_uri(),
            columns=columns,
            parquet_file_extension=None,
            index=index_column,
            dtype_backend="pyarrow",
            blocksize="1024MiB",
            storage_options={
                "key": config["minio"]["aws_access_key_id"],
                "secret": config["minio"]["aws_secret_access_key"],
                "client_kwargs": {
                    "endpoint_url": config["minio"]["endpoint_url"],
                },
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            },
        )

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        pass

    def load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        logging.info(f"Loading {self.__class__.__name__} from {self.path.as_uri()}")
        t = time.monotonic()
        ret = self._load_train_test(train_idx, test_idx, train_time_end)
        logging.info(f"Loaded {self.__class__.__name__} in {time.monotonic() - t:.3f} seconds")
        return ret
