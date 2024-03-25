import pandas as pd
import pytest

from juniper.validation.time_series_split import TimeSeriesSplit


def test_time_series_split_with_valid_input(outcomes):
    time_series_split = TimeSeriesSplit(pd.Timedelta(days=30), 3)
    splits = list(time_series_split.split(outcomes))
    assert len(splits) == 3
    assert splits[-1][2] == pd.Timestamp("2021-12-31") - pd.Timedelta(days=150)
    assert max(splits[-1][1]) == 364
    expected = [
        (
            pd.Index(range(365 - 150 - 60)),
            pd.Index(range(365 - 150 - 60 - 1, 365 - 60)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=210),
        ),
        (
            pd.Index(range(365 - 150 - 30)),
            pd.Index(range(365 - 150 - 30 - 1, 365 - 30)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=180),
        ),
        (
            pd.Index(range(365 - 150)),
            pd.Index(range(365 - 150 - 1, 365)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=150),
        ),
    ]
    for actual, exp in zip(splits, expected):
        assert actual[0].equals(exp[0])
        assert actual[1].equals(exp[1])
        assert actual[2] == exp[2]


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
