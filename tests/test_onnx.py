from unittest.mock import patch

from juniper.common.data_type import FeatureType
from juniper.common.export import to_onnx
from juniper.common.setup import load_config
from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.preprocessor.preprocessor import get_preprocessor


def test_onnx_export(config):
    config = load_config()
    enabled_feature_types = [
        FeatureType.NUMERIC,
        FeatureType.CATEGORICAL,
        FeatureType.BOOLEAN,
        # FeatureType.TIMESTAMP,
        # FeatureType.ARRAY,
    ]
    config["data_sources"]["feature_store"]["enabled_feature_types"] = enabled_feature_types
    with patch("juniper.data_loading.feature_store.load_config", return_value=config):
        feature_store = LocalParquetFeatureStore()
        column_transformer = get_preprocessor(feature_store)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    model_onnx = to_onnx(column_transformer, "test")
    assert model_onnx is not None
