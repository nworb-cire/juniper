import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
from s3path import S3Path

from juniper.common.setup import load_config
from juniper.data_loading.data_source import S3DataSource


class BaseOutcomes(S3DataSource, ABC):
    metadata: pd.Series

    def __init__(self, path: S3Path = None):
        config = load_config()
        if path is None:
            path = config["data_sources"]["outcomes"]["location"]
        self.timestamp_column = config["data_sources"]["outcomes"]["timestamp_column"]
        super().__init__(path=path)

    def index_range(self, start: datetime | None, end: datetime | None) -> pd.Index:
        if start is None:
            start = self.min_timestamp()
        if end is None:
            end = self.max_timestamp()
        return self.metadata[(self.metadata >= start) & (self.metadata < end)].index

    def min_timestamp(self) -> datetime:
        return self.metadata.min()

    def max_timestamp(self) -> datetime:
        return self.metadata.max()

    @abstractmethod
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: datetime):
        raise NotImplementedError


class PivotedOutcomes(BaseOutcomes):
    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: datetime):
        df = df.filter(lambda row: row[self.timestamp_column] < train_time_end)
        return df

    def load(self, idx, train_time_end: datetime | None = None):
        logging.info(f"Loading outcomes from {self.path.as_uri()}")
        t = time.monotonic()
        df = self.read_parquet()
        df.filter(lambda row: row[self.index_column] in idx)
        if train_time_end is not None:
            df = self.filter_training_outcomes(df, train_time_end)
        logging.info(f"Loaded outcomes in {(time.monotonic() - t):.3f} s")
        return df

    def get_metadata(self):
        logging.info(f"Loading outcomes metadata from {self.path.as_uri()}")
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
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        raise NotImplementedError


class StandardOutcomes(BaseOutcomes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = load_config()
        self.binary_outcomes_list = tuple(config["data_sources"]["outcomes"]["binary_outcomes_list"])

    def filter_training_outcomes(self, df: pd.DataFrame, train_time_end: datetime):
        _offsets = set(c.split("_")[-1] for c in df.columns if c.startswith(self.binary_outcomes_list))
        for offset in _offsets:
            offset = int(offset)
            cols = [c for c in df.columns if c.endswith(f"_{offset}")]
            delta_holdout_date = train_time_end - pd.Timedelta(days=offset)
            for col in cols:
                df[col] = df[col].mask(df[self.timestamp_column] >= delta_holdout_date, None)
        return df

    def _load_train_test(
        self, train_idx: pd.Index, test_idx: pd.Index = None, train_time_end: datetime = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        df = self.read_parquet(
            filters=[(self.index_column, "in", train_idx.union(test_idx).tolist())],
        )
        columns = [c for c in df.columns if c.startswith(self.binary_outcomes_list) and re.match(r"\w+_\d{1,4}$", c)]
        df = df[columns + [self.timestamp_column]]
        train = df.reindex(train_idx)
        train = self.filter_training_outcomes(train, train_time_end).drop(columns=[self.timestamp_column])
        if test_idx is not None:
            test = df.reindex(test_idx)
        else:
            test = None
        return train, test

    def get_metadata(self) -> pd.Series:
        df = self.read_parquet(columns=[self.timestamp_column])
        df = df[self.timestamp_column]
        df = df.dropna()
        # df = df.sort_values()
        self.metadata = df
        return df
