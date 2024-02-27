import pytest
from sklearn.compose import ColumnTransformer

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


@pytest.mark.parametrize("column", ["numeric", "categorical", "boolean", "timestamp"])
def test_fit_preprocessor(column, feature_store):
    column_transformer = get_preprocessor(feature_store)
    _, preprocessor, columns = next(filter(lambda x: x[0] == column, column_transformer.transformers))
    assert preprocessor is not None
    df = feature_store.read_parquet()
    Xt = preprocessor.fit_transform(df[columns])
    assert Xt is not None
