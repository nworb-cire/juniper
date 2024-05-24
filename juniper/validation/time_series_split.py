import logging
from collections.abc import Generator

import pandas as pd

from juniper.data_loading.outcomes import BaseOutcomes


class TimeSeriesSplit:
    def __init__(
        self,
        timedelta: pd.Timedelta,
        n_splits: int,
        gap_days: int = 0,
        holdout_time: int = 150,
        include_last: bool = False,
    ):
        if n_splits <= 0:
            raise ValueError("n_splits must be a positive integer")
        self.timedelta = timedelta
        self.n_splits = n_splits
        self.gap = gap_days
        # self.scores = []
        self.holdout_time = holdout_time
        if include_last:
            raise NotImplementedError

    def split(
        self, outcomes: BaseOutcomes, end_ts: pd.Timestamp = None
    ) -> Generator[tuple[pd.Index, pd.Index, pd.Timestamp], None, None]:
        if end_ts is None:
            end_ts = outcomes.max_timestamp()
        for i in range(self.n_splits):
            holdout_time_end = end_ts - (self.n_splits - i - 1) * self.timedelta
            holdout_time_begin = holdout_time_end - pd.Timedelta(days=self.holdout_time)
            train_time_end = holdout_time_begin - pd.Timedelta(days=self.gap)
            train_idx = outcomes.index_range(None, train_time_end)
            test_idx = outcomes.index_range(holdout_time_begin, holdout_time_end)
            logging.info("#" * 10 + f" Time Series CV Split #{i+1}/{self.n_splits} " + "#" * 10)
            logging.info(f"-----> {train_time_end} | {holdout_time_begin} <---> {holdout_time_end}")
            logging.info(f" Train size: {len(train_idx)}, val size: {len(test_idx)}")
            yield train_idx, test_idx, train_time_end

    # def add_score(self, score):
    #     # TODO: Integrate metric logging with an experiment tracking system
    #     self.scores.append(score)
