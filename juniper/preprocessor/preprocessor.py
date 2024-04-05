import logging
from functools import partial

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
    prefix: str = "",
) -> ColumnTransformer:
    if schema is None:
        schema = feature_store.schema
    metadata = feature_store.get_feature_metadata(schema)

    transformers = []
    if columns := metadata.get(FeatureType.NUMERIC):
        if numeric_pipeline is None:
            numeric_pipeline = get_default_numeric_pipeline(columns)
        transformers.append((f"{prefix}numeric", numeric_pipeline, columns))

    if columns := metadata.get(FeatureType.CATEGORICAL):
        if categorical_pipeline is None:
            categorical_pipeline = get_default_categorical_pipeline(columns)
        transformers.append((f"{prefix}categorical", categorical_pipeline, columns))

    if columns := metadata.get(FeatureType.BOOLEAN):
        if boolean_pipeline is None:
            boolean_pipeline = get_default_boolean_pipeline(columns)
        transformers.append((f"{prefix}boolean", boolean_pipeline, columns))

    if columns := metadata.get(FeatureType.TIMESTAMP):
        if timestamp_pipeline is None:
            timestamp_pipeline = get_default_timestamp_pipeline(columns)
        transformers.append((f"{prefix}timestamp", timestamp_pipeline, columns))

    if columns := metadata.get(FeatureType.ARRAY):
        config = load_config()
        for column in columns:
            feature_metadata = config["data_sources"]["feature_store"].get("feature_meta", {}).get(column, {})
            try:
                transformer = ColumnNormalizer(
                    field=schema.field(column),
                    preprocessor_factory=partial(
                        get_preprocessor, feature_store=feature_store, prefix=f"{prefix}{column}."
                    ),
                    record_path=feature_metadata.get("record_path"),
                    meta=feature_metadata.get("meta"),
                )
                transformers.append((column, transformer, [column]))
            except ValueError as e:
                logging.warning(f"Error creating ColumnNormalizer for {column}: {e}")
                continue

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1,
        verbose=True,
        verbose_feature_names_out=False,
    )
    if len(transformers) == 0:
        raise ValueError("No transformers found")
    logging.debug(f"Preprocessor initialized with {len(transformers)} transformers")
    for transformer in transformers:
        logging.debug(f"Transformer: {transformer[0]} ({len(transformer[2])} columns)")
    column_transformer.set_output(transform="pandas")
    return column_transformer
