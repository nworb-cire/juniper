import logging
import re
import time
from abc import ABC, abstractmethod

import pandas as pd
from s3path import S3Path

from juniper.common.setup import load_config
from juniper.data_loading.data_source import S3ParquetDataSource, BaseDataSource, LocalDataSource, ParquetDataSource


class BaseOutcomes(BaseDataSource, ABC):
    config_location = "outcomes"
    metadata: pd.Series

    def __init__(self, path: S3Path = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["outcomes"]["location"]
        self.path = path
        super().__init__()

    def index_range(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Index:
        if start is None:
            start = self.min_timestamp()
        if end is None:
            end = self.max_timestamp()
        return self.metadata[(self.metadata >= start) & (self.metadata <= end)].index.unique()

    def min_timestamp(self) -> pd.Timestamp:
        return self.metadata.min()

    def max_timestamp(self) -> pd.Timestamp:
        return self.metadata.max()

    @abstractmethod
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        raise NotImplementedError


class PivotedOutcomes(BaseOutcomes, S3ParquetDataSource):
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        df = df.filter(lambda row: row[self.timestamp_column] < train_time_end)
        return df

    def load(self, idx, train_time_end: pd.Timestamp | None = None):
        logging.info(f"Loading outcomes from {self._path_str}")
        t = time.monotonic()
        df = self.read_parquet()
        df.filter(lambda row: row[self.index_column] in idx)
        if train_time_end is not None:
            df = self.filter_training_outcomes(df, train_time_end)
        logging.info(f"Loaded outcomes in {(time.monotonic() - t):.3f} s")
        return df

    def get_metadata(self):
        logging.info(f"Loading outcomes metadata from {self._path_str}")
        t = time.monotonic()
        df = self.read_parquet(columns=[self.index_column, self.timestamp_column]).to_pandas()
        df = df[~df[self.timestamp_column].isna()]
        df = df.drop_duplicates(subset=[self.index_column, self.timestamp_column])
        df = df.set_index(self.index_column)
        df = df.dropna()
        # df = df.sort_values(self.timestamp_column.value)
        logging.info(f"Loaded outcomes metadata in {(time.monotonic() - t):.3f} ms")
        return df

    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: pd.Timestamp = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        raise NotImplementedError


class StandardOutcomes(BaseOutcomes, ParquetDataSource, ABC):
    def __init__(self, *args, **kwargs):
        config = load_config()
        self.binary_outcomes_list = tuple(config["data_sources"]["outcomes"]["binary_outcomes_list"])
        super().__init__(*args, **kwargs)

    def _get_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith(self.binary_outcomes_list) and re.match(r"\w+_\d{1,4}$", c)]

    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: pd.Timestamp):
        all_cols = self._get_columns(df)
        _offsets = set(c.split("_")[-1] for c in all_cols)
        for offset in _offsets:
            offset = int(offset)
            cols = [c for c in df.columns if c.endswith(f"_{offset}")]
            delta_holdout_date = train_time_end - pd.Timedelta(days=offset)
            for col in cols:
                df[col] = df[col].mask(df[self.timestamp_column] >= delta_holdout_date, None)
        return df.dropna(how="all", subset=all_cols)

    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: pd.Timestamp = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        df = self.read_parquet(
            filters=[(self.index_column, "in", train_idx.union(test_idx).tolist())],
        )
        df = df[self._get_columns(df) + [self.timestamp_column]]
        df = df.sort_values(self.timestamp_column)
        df = df[~df.index.duplicated(keep="last")]
        train = df.reindex(train_idx)
        train = self.filter_training_outcomes(train, train_time_end).drop(columns=[self.timestamp_column])
        if test_idx is not None:
            test = df.reindex(test_idx).drop(columns=[self.timestamp_column])
        else:
            test = None
        return train, test

    def get_metadata(self) -> pd.Series:
        df = self.read_parquet()
        columns = self._get_columns(df)
        df = df[columns + [self.timestamp_column]]
        df = df.dropna(subset=columns, how="all")
        df = df[self.timestamp_column]
        df = df.dropna()
        # df = df.sort_values()
        return df


class LocalStandardOutcomes(StandardOutcomes, LocalDataSource):
    pass
