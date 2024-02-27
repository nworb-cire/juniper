import pandas as pd


def test_read_parquet(feature_store):
    df = feature_store.read_parquet()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == feature_store.index_column
    idx = pd.Index([1, 2, 3, 4], name="id")
    pd.testing.assert_index_equal(df.index, idx)
