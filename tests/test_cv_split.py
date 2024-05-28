import pandas as pd
import pytest

from juniper.validation.time_series_split import TimeSeriesSplit


def test_time_series_split_with_valid_input(outcomes):
    lag = 150
    split = 30
    time_series_split = TimeSeriesSplit(pd.Timedelta(days=split), 3)
    splits = list(time_series_split.split(outcomes))
    assert len(splits) == 3
    assert splits[-1][2] == pd.Timestamp("2021-12-31") - pd.Timedelta(days=lag)
    assert max(splits[-1][1]) == 364
    expected = [
        (
            pd.Index(range(365 - lag - 2 * split)),
            pd.Index(range(365 - lag - 2 * split, 365 - 2 * split)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=lag + 2 * split),
        ),
        (
            pd.Index(range(365 - lag - split)),
            pd.Index(range(365 - lag - split, 365 - split)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=lag + split),
        ),
        (
            pd.Index(range(365 - lag)),
            pd.Index(range(365 - lag, 365)),
            pd.Timestamp("2021-12-31") - pd.Timedelta(days=lag),
        ),
    ]
    for actual, exp in zip(splits, expected):
        print(exp[2])
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
    splits = list(time_series_split.split(outcomes=outcomes, end_ts=end_ts))
    assert len(splits) == 3
    for split in splits:
        assert split[2] <= end_ts
    assert all(split[2] <= end_ts for split in splits)
