import pandas as pd
import pytest

from juniper.validation.time_series_split import TimeSeriesSplit


def test_time_series_split_with_valid_input(outcomes):
    time_series_split = TimeSeriesSplit(pd.Timedelta(days=30), 3)
    splits = list(time_series_split.split(outcomes))
    assert len(splits) == 3


@pytest.mark.parametrize("split", [0, -1])
def test_time_series_split_with_non_positive_splits(outcomes, split):
    with pytest.raises(ValueError):
        TimeSeriesSplit(pd.Timedelta(days=30), split)


def test_time_series_split_with_include_last(outcomes):
    with pytest.raises(NotImplementedError):
        TimeSeriesSplit(pd.Timedelta(days=30), 3, include_last=True)


def test_time_series_split_with_custom_end_ts(outcomes):
    time_series_split = TimeSeriesSplit(pd.Timedelta(days=30), 3)
    end_ts = pd.Timestamp("2021-01-01")
    splits = list(time_series_split.split(outcomes, end_ts=end_ts))
    assert len(splits) == 3
    assert all(split[2] <= end_ts for split in splits)
