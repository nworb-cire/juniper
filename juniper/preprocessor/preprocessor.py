import logging

import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.data_loading.feature_store import BaseFeatureStore
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.preprocessor.pipelines import (
    get_default_numeric_pipeline,
    get_default_categorical_pipeline,
    get_default_boolean_pipeline,
    get_default_timestamp_pipeline,
)


def get_preprocessor(
    feature_store: BaseFeatureStore,
    schema: pa.Schema = None,
    numeric_pipeline: Pipeline | None = None,
    categorical_pipeline: Pipeline | None = None,
    boolean_pipeline: Pipeline | None = None,
    timestamp_pipeline: Pipeline | None = None,
) -> ColumnTransformer:
    if schema is None:
        metadata = feature_store.metadata
    else:
        metadata = feature_store.get_feature_metadata(schema)

    transformers = []
    if columns := metadata.get(FeatureType.NUMERIC):
        if numeric_pipeline is None:
            numeric_pipeline = get_default_numeric_pipeline(columns)
        transformers.append(("numeric", numeric_pipeline, columns))

    if columns := metadata.get(FeatureType.CATEGORICAL):
        if categorical_pipeline is None:
            categorical_pipeline = get_default_categorical_pipeline(columns)
        transformers.append(("categorical", categorical_pipeline, columns))

    if columns := metadata.get(FeatureType.BOOLEAN):
        if boolean_pipeline is None:
            boolean_pipeline = get_default_boolean_pipeline(columns)
        transformers.append(("boolean", boolean_pipeline, columns))

    if columns := metadata.get(FeatureType.TIMESTAMP):
        if timestamp_pipeline is None:
            timestamp_pipeline = get_default_timestamp_pipeline(columns)
        transformers.append(("timestamp", timestamp_pipeline, columns))

    if columns := metadata.get(FeatureType.ARRAY):
        config = load_config()
        for column in columns:
            feature_metadata = config["data_sources"]["feature_store"].get("feature_meta", {}).get(column, {})
            cn = ColumnNormalizer(
                column_name=column,
                schema_in=feature_store.schema,
                record_path=feature_metadata.get("record_path"),
                meta=feature_metadata.get("meta"),
            )
            if all(field.metadata[b"usable_type"].decode() == FeatureType.UNUSABLE for field in cn.schema_out):
                logging.warning(f"Array column {column} is unusable and will be dropped")
                continue
            pipeline = Pipeline(
                steps=[
                    ("normalizer", cn),
                    ("preprocessor", get_preprocessor(feature_store, cn.schema_out)),
                ]
            )
            transformers.append((column, pipeline, [column]))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=-1)
    if len(transformers) == 0:
        raise ValueError("No transformers found")
    logging.info(f"Preprocessor initialized with {len(transformers)} transformers")
    for transformer in transformers:
        logging.info(f"Transformer: {transformer[0]} ({len(transformer[2])} columns)")
    return column_transformer
