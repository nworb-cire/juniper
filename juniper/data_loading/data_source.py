import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from s3path import S3Path

from juniper.common.setup import load_config


class BaseDataSource(ABC):
    index_column: str
    timestamp_column: str
    config_location: str

    def __init__(self):
        config = load_config()
        self.index_column = config["data_sources"]["index_column"]
        self.timestamp_column = config["data_sources"][self.config_location]["timestamp_column"]
        self.metadata = self.get_metadata()

    @abstractmethod
    def get_metadata(self):
        pass

    @property
    @abstractmethod
    def _path_str(self) -> str:
        pass

    @abstractmethod
    def _load_train_test(
        self,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        train_time_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        pass

    def load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index | None = None, train_time_end: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        logging.info(f"Loading {self.__class__.__name__} from {self._path_str}")
        t = time.monotonic()
        ret = self._load_train_test(train_idx, test_idx, train_time_end)
        logging.info(f"Loaded {self.__class__.__name__} in {time.monotonic() - t:.3f} seconds")
        msg = f"  Train size: {len(ret[0])}"
        if ret[1] is not None:
            msg += f", test size: {len(ret[1])}"
        logging.info(msg)
        return ret

    @abstractmethod
    def read_timestamps(self) -> pd.Series:
        pass


class ParquetDataSource(BaseDataSource, ABC):
    path: Path

    @abstractmethod
    def read_parquet(
        self,
        path: Path | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
    ) -> pd.DataFrame:
        pass


class LocalDataSource(ParquetDataSource, ABC):
    @property
    def _path_str(self) -> str:
        return str(self.path.absolute())

    def read_parquet(
        self,
        path: Path | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
    ) -> pd.DataFrame:
        if path is None:
            path = self.path
        if columns is not None and self.index_column not in columns:
            columns.append(self.index_column)
        df = pd.read_parquet(path, columns=columns, filters=filters)
        return df.set_index(self.index_column)

    def read_timestamps(self) -> pd.Series:
        return self.read_parquet(columns=[self.timestamp_column])[self.timestamp_column]


class S3ParquetDataSource(ParquetDataSource, ABC):
    path: S3Path

    @property
    def _path_str(self) -> str:
        return self.path.as_uri()

    def read_parquet(
        self,
        path: S3Path | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
    ) -> pd.DataFrame:
        if path is None:
            path = self.path
        if columns is not None and self.index_column not in columns:
            columns.append(self.index_column)
        config = load_config()
        return pd.read_parquet(
            path.as_uri(),
            columns=columns,
            dtype_backend="pyarrow",
            filters=filters,
            storage_options={
                "key": config["minio"]["aws_access_key_id"],
                "secret": config["minio"]["aws_secret_access_key"],
                "client_kwargs": {
                    "endpoint_url": config["minio"]["endpoint_url"],
                },
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            },
        )


class SqlDataSource(BaseDataSource, ABC):
    connection_str: str

    @property
    def _path_str(self) -> str:
        return self.connection_str.split("/")[-1]

    def read_sql(self, query: str, params=None) -> pd.DataFrame:
        return pd.read_sql(query, self.connection_str, params=params)
