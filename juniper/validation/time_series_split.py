import logging
from collections.abc import Generator

import pandas as pd

from juniper.data_loading.feature_store import BaseFeatureStore
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
            holdout_time_begin = holdout_time_end - pd.Timedelta(days=self.holdout_time)
            train_time_end = holdout_time_begin - pd.Timedelta(days=self.gap)
            train_idx = ts[ts <= train_time_end].index
            test_idx = ts[(ts > holdout_time_begin) & (ts <= holdout_time_end)].index
            logging.info("#" * 10 + f" Time Series CV Split #{i+1}/{self.n_splits} " + "#" * 10)
            logging.info(f"-----> {train_time_end} | {holdout_time_begin} <---> {holdout_time_end}")
            logging.info(f" Train size: {len(train_idx)}, val size: {len(test_idx)}")
            yield train_idx, test_idx, train_time_end

    # def add_score(self, score):
    #     # TODO: Integrate metric logging with an experiment tracking system
    #     self.scores.append(score)
