import logging
import re
import time
from abc import ABC, abstractmethod

import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import dataset as ds
from pyarrow import parquet as pq
from s3path import S3Path

from juniper.common.setup import load_config
from juniper.data_loading.data_source import BaseDataSource, LocalDataSource, ParquetDataSource


class BaseOutcomes(BaseDataSource, ABC):
    config_location = "outcomes"
    metadata: pd.Series

    def __init__(self, path: S3Path | None = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["outcomes"]["location"]
        self.path = path
        super().__init__()

    @abstractmethod
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        raise NotImplementedError


class BasePivotedOutcomes(BaseOutcomes, ParquetDataSource, ABC):
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        df = df.filter(lambda row: row[self.timestamp_column] < train_time_end)
        return df

    def get_metadata(self):
        logging.info(f"Loading outcomes metadata from {self._path_str}")
        t = time.monotonic()
        df = self.read_parquet(columns=[self.index_column, self.timestamp_column])
        df = df[~df[self.timestamp_column].isna()]
        df = df.reset_index(drop=False)
        df = df.drop_duplicates(subset=[self.index_column, self.timestamp_column])
        df = df.set_index(self.index_column)
        df = df.dropna()
        # df = df.sort_values(self.timestamp_column.value)
        logging.info(f"Loaded outcomes metadata in {(time.monotonic() - t):.3f} ms")
        return df

    def _load_train_test(
        self,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        train_time_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        config = load_config()
        outcomes_columns = config["data_sources"]["outcomes"]["binary_outcomes_list"]
        pivot_column = config["data_sources"]["outcomes"]["pivot_column"]
        filters = ~ds.field(self.index_column).is_null()
        if train_idx is not None:
            filters &= pc.is_in(ds.field(self.index_column), pa.array(train_idx))
        train = self.read_parquet(
            filters=filters,
            columns=[outcomes_columns + [self.index_column, self.timestamp_column, pivot_column]],
        )
        train = self.filter_training_outcomes(train, train_time_end)
        train = train.pivot(index=self.index_column, columns=pivot_column, values=outcomes_columns)
        if test_idx is not None:
            filters = ~ds.field(self.index_column).is_null() & pc.is_in(ds.field(self.index_column), pa.array(test_idx))
            test = self.read_parquet(
                filters=filters,
                columns=[outcomes_columns + [self.index_column, self.timestamp_column, pivot_column]],
            )
            test = test.pivot(index=self.index_column, columns=pivot_column, values=outcomes_columns)
        else:
            test = None
        return train, test


class LocalPivotedOutcomes(BasePivotedOutcomes, LocalDataSource):
    pass


class StandardOutcomes(BaseOutcomes, ParquetDataSource, ABC):
    def __init__(self, *args, **kwargs):
        config = load_config()
        self.binary_outcomes_list = tuple(config["data_sources"]["outcomes"]["binary_outcomes_list"])
        super().__init__(*args, **kwargs)

    def _get_columns(self, columns: list[str] | None = None) -> list[str]:
        if columns is None:
            columns = self.all_columns
        return [c for c in columns if c.startswith(self.binary_outcomes_list) and re.match(r"\w+_\d{1,4}$", c)]

    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        all_cols = self._get_columns()
        _offsets = set(c.split("_")[-1] for c in all_cols)
        for off in _offsets:
            offset = int(off)
            cols = [c for c in df.columns if c.endswith(f"_{offset}")]
            delta_holdout_date = train_time_end - pd.Timedelta(days=offset)
            for col in cols:
                df[col] = df[col].mask(df[self.timestamp_column] >= delta_holdout_date, None)
        return df.dropna(how="all", subset=all_cols)

    def _load_train_test(
        self,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        train_time_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        filters = ~ds.field(self.index_column).is_null()
        if train_idx is not None:
            filters &= pc.is_in(ds.field(self.index_column), pa.array(train_idx))
        train = self.read_parquet(filters=filters, columns=self._get_columns() + [self.timestamp_column])
        train = self.filter_training_outcomes(train, train_time_end).drop(columns=[self.timestamp_column])
        if test_idx is not None:
            filters = ~ds.field(self.index_column).is_null() & pc.is_in(ds.field(self.index_column), pa.array(test_idx))
            test = self.read_parquet(filters=filters, columns=self._get_columns() + [self.timestamp_column])
        else:
            test = None
        return train, test

    def get_metadata(self) -> pd.Series:
        self.all_columns = pq.read_schema(self.path).names
        df = self.read_parquet()
        columns = self._get_columns()
        df = df[columns + [self.timestamp_column]]
        df = df.dropna(subset=columns, how="all")
        df = df[self.timestamp_column]
        df = df.dropna()
        # df = df.sort_values()
        return df


class LocalStandardOutcomes(StandardOutcomes, LocalDataSource):
    pass
