import logging
from functools import partial, reduce

import onnx
import pyarrow as pa
import sklearn.compose
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline

from juniper.common.data_type import FeatureType
from juniper.common.export import get_onnx_types, merge_models
from juniper.data_loading.feature_types import get_feature_types
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.preprocessor.pipelines import (
    get_default_numeric_pipeline,
    get_default_categorical_pipeline,
    get_default_boolean_pipeline,
    get_default_timestamp_pipeline,
)


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    def __init__(
        self,
        schema: pa.Schema,
        numeric_pipeline: Pipeline | None = None,
        categorical_pipeline: Pipeline | None = None,
        boolean_pipeline: Pipeline | None = None,
        timestamp_pipeline: Pipeline | None = None,
        prefix: str = "",
        n_jobs: int = -1,
    ):
        metadata = get_feature_types(schema)

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
            for column in columns:
                try:
                    cn = ColumnNormalizer(
                        field=schema.field(column),
                        preprocessor_factory=partial(ColumnTransformer, schema=schema, prefix=f"{prefix}{column}."),
                    )
                    transformers.append((column, cn, [column]))
                except ValueError as e:
                    logging.warning(f"Error creating ColumnNormalizer for {column}: {e}")
                    continue

        super().__init__(
            transformers=transformers,
            remainder="drop",
            n_jobs=n_jobs,
            verbose=True,
            verbose_feature_names_out=False,
        )
        if len(transformers) == 0:
            raise ValueError("No transformers found")
        logging.debug(f"Preprocessor initialized with {len(transformers)} transformers")
        for transformer in transformers:
            logging.debug(f"Transformer: {transformer[0]} ({len(transformer[2])} columns)")
        self.set_output(transform="pandas")

    def to_onnx(self, name: str | None = None) -> onnx.ModelProto:
        transformers = []
        sub_transformers = []
        for name_, t, cols in self.transformers_:
            if isinstance(t, ColumnNormalizer):
                sub_transformers.append(t.column_transformer.to_onnx(name_))
            elif name_ == "remainder":
                continue
            else:
                transformers.append((name_, t, cols))
        ct_out = sklearn.compose.ColumnTransformer(transformers, remainder="drop")
        ct_out.transformers_ = transformers

        model_onnx = convert_sklearn(
            model=ct_out,
            name=name if name is not None else "base",
            initial_types=get_onnx_types(ct_out),
            naming=name + "_" if name is not None else "",
            target_opset=17,
        )
        onnx.checker.check_model(model_onnx, full_check=True)
        # rename output
        assert len(model_onnx.graph.output) == 1
        output_node_name = model_onnx.graph.output[0].name
        renamed_node_name = "features" if name is None else name
        model_onnx.graph.output[0].name = renamed_node_name
        for node in model_onnx.graph.node:
            for i in range(len(node.output)):
                if node.output[i] == output_node_name:
                    node.output[i] = renamed_node_name
        # merge subgraphs
        model_onnx = reduce(partial(merge_models, io_map=[]), sub_transformers, model_onnx)
        return model_onnx

    def _get_param_names(cls):
        return sklearn.compose.ColumnTransformer._get_param_names()
