import logging

import pyarrow as pa
from sklearn.pipeline import Pipeline

from src.common.setup import load_config
from src.preprocessor.dask.column_normalizer import ColumnNormalizer
from src.preprocessor.dask.column_transformer import ColumnTransformer
from src.preprocessor.metadata import FeatureStoreMetadata
from src.preprocessor.pipelines import get_default_numeric_pipeline, get_default_categorical_pipeline, \
    get_default_boolean_pipeline, get_default_timestamp_pipeline


def get_preprocessor(
    schema: pa.Schema,
    numeric_pipeline: Pipeline | None = None,
    categorical_pipeline: Pipeline | None = None,
    boolean_pipeline: Pipeline | None = None,
    timestamp_pipeline: Pipeline | None = None,
):
    metadata = FeatureStoreMetadata(schema)
    
    transformers = []
    if columns := metadata.numeric_columns:
        if numeric_pipeline is None:
            numeric_pipeline = get_default_numeric_pipeline(columns)
        transformers.append(("numeric", numeric_pipeline, columns))

    if columns := metadata.categorical_columns:
        if categorical_pipeline is None:
            categorical_pipeline = get_default_categorical_pipeline(columns)
        transformers.append(("categorical", categorical_pipeline, columns))

    if columns := metadata.boolean_columns:
        if boolean_pipeline is None:
            boolean_pipeline = get_default_boolean_pipeline(columns)
        transformers.append(("boolean", boolean_pipeline, columns))

    if columns := metadata.timestamp_columns:
        if timestamp_pipeline is None:
            timestamp_pipeline = get_default_timestamp_pipeline(columns)
        transformers.append(("timestamp", timestamp_pipeline, columns))

    if columns := metadata.array_columns:
        config = load_config()
        for column in columns:
            feature_metadata = config["data_sources"]["feature_store"]["feature_meta"].get(column, {})
            cn = ColumnNormalizer(
                column_name=column,
                schema_in=schema,
                record_path=feature_metadata.get("record_path"),
                meta=feature_metadata.get("meta"),
            )
            pipeline = Pipeline(steps=[
                ("normalizer", cn),
                ("preprocessor", get_preprocessor(cn.schema_out)),
            ])
            transformers.append((column, pipeline, [column]))

    column_transformer = ColumnTransformer(transformers=transformers, remainder='drop')
    logging.info(f"Preprocessor initialized with {len(transformers)} transformers")
    for transformer in transformers:
        logging.info(f"Transformer: {transformer[0]} ({len(transformer[2])} columns)")
    return column_transformer
