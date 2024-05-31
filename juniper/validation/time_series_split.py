import logging
from collections.abc import Generator

import pandas as pd

from juniper.data_loading.feature_store import BaseFeatureStore
from juniper.data_loading.outcomes import BaseOutcomes


DEFAULT_HOLDOUT_TIME = pd.Timedelta(days=150)


class TimeSeriesSplit:
    def __init__(
        self,
        timedelta: pd.Timedelta,
        n_splits: int,
        gap_days: int = 0,
        holdout_time: pd.Timedelta = DEFAULT_HOLDOUT_TIME,
        include_last: bool = False,
    ):
        """
        Perform a time series cross validation split on a dataset. The dataset is split into n_splits, with each split
        having a holdout time of holdout_time. The gap_days parameter specifies the number of days between the end of
        the training set and the beginning of the holdout set. 
        
        Example: (n_splits=3, holdout_time=3, gap_days=1, timedelta=1, include_last=False)
        Split 1: TTT HHH
        Split 2: TTTT HHH
        Split 3: TTTTT HHH
        
        :param timedelta: Amount of time to advance the holdout window each split.
        :param n_splits: How many splits to perform. 
        :param gap_days: Amount of time to wait between the end of the training set and the beginning of the holdout set.
        :param holdout_time: Duration of the holdout set.
        :param include_last: If True, the final training set will include the entire dataset with an empty holdout set.
        """
        if n_splits <= 0:
            raise ValueError("n_splits must be a positive integer")
        self.timedelta = timedelta
        self.include_last = include_last
        self.n_splits = n_splits if not include_last else n_splits - 1
        self.gap = gap_days
        # self.scores = []
        self.holdout_time = holdout_time

    def split(
        self,
        features: BaseFeatureStore | None = None,
        outcomes: BaseOutcomes | None = None,
        end_ts: pd.Timestamp | None = None,
    ) -> Generator[tuple[pd.Index, pd.Index, pd.Timestamp], None, None]:
        if features is None and outcomes is None:
            raise ValueError("Either features or outcomes must be provided")
        elif features is None:
            assert outcomes is not None
            ts = outcomes.read_timestamps()
        elif outcomes is None:
            assert features is not None
            ts = features.read_timestamps()
        else:
            feature_ts = features.read_timestamps()
            outcomes_ts = outcomes.read_timestamps()
            ts = feature_ts[feature_ts.index.intersection(outcomes_ts.index)]

        if end_ts is None:
            end_ts = ts.max()
        else:
            end_ts = min(end_ts, ts.max())

        for i in range(self.n_splits):
            holdout_time_end = end_ts - (self.n_splits - i - 1) * self.timedelta
            holdout_time_begin = holdout_time_end - self.holdout_time
            train_time_end = holdout_time_begin - pd.Timedelta(days=self.gap)
            train_idx = ts[ts <= train_time_end].index
            test_idx = ts[(ts > holdout_time_begin) & (ts <= holdout_time_end)].index
            logging.info("#" * 10 + f" Time Series CV Split #{i+1}/{self.n_splits} " + "#" * 10)
            logging.info(f"-----> {train_time_end} | {holdout_time_begin} <---> {holdout_time_end}")
            logging.info(f" Train size: {len(train_idx)}, val size: {len(test_idx)}")
            yield train_idx, test_idx, train_time_end
        if self.include_last:
            yield ts.index, pd.Index([]), end_ts

    # def add_score(self, score):
    #     # TODO: Integrate metric logging with an experiment tracking system
    #     self.scores.append(score)
