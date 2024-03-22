import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
from s3path import S3Path

from juniper.common.setup import load_config


class BaseDataSource(ABC):
    index_column: str
    timestamp_column: str

    def __init__(self, path: Path):
        self.path = path
        config = load_config()
        self.index_column = config["data_sources"]["index_column"]
        self.get_metadata()

    @property
    @abstractmethod
    def _path_str(self) -> str:
        pass

    @abstractmethod
    def read_parquet(
        self,
        path: Path = None,
        columns: list[str] = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
    ) -> pd.DataFrame:
        pass

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
        logging.info(f"Loading {self.__class__.__name__} from {self._path_str}")
        t = time.monotonic()
        ret = self._load_train_test(train_idx, test_idx, train_time_end)
        logging.info(f"Loaded {self.__class__.__name__} in {time.monotonic() - t:.3f} seconds")
        logging.info(f"  Train size: {len(ret[0])}, test size: {len(ret[1])}")
        return ret


class LocalDataSource(BaseDataSource, ABC):
    def __init__(self, path: Path):
        super().__init__(path)

    @property
    def _path_str(self) -> str:
        return str(self.path.absolute())

    def read_parquet(
        self, path: Path = None, columns: list[str] = None, filters: list[tuple] | list[list[tuple]] | None = None
    ) -> pd.DataFrame:
        if path is None:
            path = self.path
        df = pd.read_parquet(path, columns=columns, filters=filters)
        return df.set_index(self.index_column)


class S3DataSource(BaseDataSource, ABC):
    def __init__(self, path: S3Path):
        super().__init__(path)

    @property
    def _path_str(self) -> str:
        return self.path.as_uri()

    def read_parquet(
        self,
        path: S3Path = None,
        columns: list[str] = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
    ) -> pd.DataFrame:
        if path is None:
            path = self.path
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
