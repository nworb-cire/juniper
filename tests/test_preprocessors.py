import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from juniper.common.data_type import FeatureType
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.preprocessor.preprocessor import get_preprocessor


def test_get_preprocessor(feature_store):
    preprocessor = get_preprocessor(feature_store)
    assert preprocessor is not None
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 5
    assert preprocessor.transformers[0][0] == "numeric"
    assert preprocessor.transformers[1][0] == "categorical"
    assert preprocessor.transformers[2][0] == "boolean"
    assert preprocessor.transformers[3][0] == "timestamp"
    assert preprocessor.transformers[4][0] == "arr"


@pytest.mark.parametrize(
    "feature_type, expected",
    [
        (FeatureType.NUMERIC, np.array([[0.5154639, 0.0, -0.5154639, 0.0], [0.0, 0.0, 0.0, 1.0]])),
        (FeatureType.CATEGORICAL, [1.0, 0.0, 2.0, 3.0]),
        (FeatureType.BOOLEAN, [1.0, 1.0, -1.0, 0.0]),
        (FeatureType.TIMESTAMP, [-1.0, -1 / 3, 1 / 3, 1.0]),
    ],
)
def test_fit_preprocessor(feature_type, expected, feature_store):
    column_transformer = get_preprocessor(feature_store)
    _, preprocessor, columns = next(filter(lambda x: x[0] == feature_type, column_transformer.transformers))
    assert preprocessor is not None
    df = feature_store.read_parquet()
    Xt = preprocessor.fit_transform(df[columns])
    assert Xt is not None
    assert isinstance(Xt, np.ndarray)
    assert np.allclose(Xt.T, expected)


def test_get_array_metadata(feature_store):
    cn = ColumnNormalizer(column_name="arr", schema_in=feature_store.schema)
    assert cn.schema_out is not None
    metadata = feature_store.get_feature_metadata(cn.schema_out)
    assert metadata is not None
    assert metadata == {
        FeatureType.NUMERIC: ["arr.a", "arr.b"],
    }


def test_normalize_array(feature_store):
    df = feature_store.read_parquet()[["arr"]]
    assert df is not None
    cn = ColumnNormalizer(column_name="arr", schema_in=feature_store.schema)
    Xt = cn.fit_transform(df)
    assert Xt is not None
    assert isinstance(Xt, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "arr.a": [1.0, 3.0, 5.0, 7.0],
            "arr.b": [2.0, 4.0, 6.0, 8.0],
        },
        index=pd.Index([1, 1, 4, 4], name="id"),
    )
    pd.testing.assert_frame_equal(Xt.astype(float), expected.astype(float))


def test_fit_array_preprocessor(feature_store):
    column_transformer = get_preprocessor(feature_store)
    _, preprocessor, columns = next(filter(lambda x: x[0] == "arr", column_transformer.transformers))
    assert preprocessor is not None
    df = feature_store.read_parquet()
    Xt = preprocessor.fit_transform(df[columns])
    assert Xt is not None
    expected = np.array(
        [[-0.5102041, -0.5102041], [-0.17006803, -0.17006803], [0.17006803, 0.17006803], [0.5102041, 0.5102041]]
    )
    assert np.allclose(Xt, expected)
