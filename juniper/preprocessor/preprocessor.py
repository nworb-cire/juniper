import logging
from functools import partial, reduce

import onnx
import pyarrow as pa
import sklearn.compose
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline

from juniper.common.component import ModelComponent
from juniper.common.data_type import FeatureType
from juniper.common.export import get_onnx_types, merge_models
from juniper.common.setup import load_config
from juniper.data_loading.feature_store import BaseFeatureStore
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.preprocessor.pipelines import (
    get_default_numeric_pipeline,
    get_default_categorical_pipeline,
    get_default_boolean_pipeline,
    get_default_timestamp_pipeline,
)
from juniper.modeling.metrics import EvalMetrics


class ColumnTransformer(sklearn.compose.ColumnTransformer, ModelComponent):
    def __init__(
        self,
        feature_store: BaseFeatureStore,
        schema: pa.Schema | None = None,
        numeric_pipeline: Pipeline | None = None,
        categorical_pipeline: Pipeline | None = None,
        boolean_pipeline: Pipeline | None = None,
        timestamp_pipeline: Pipeline | None = None,
        prefix: str = "",
        n_jobs: int = -1,
    ):
        if schema is None:
            schema = feature_store.schema
        metadata = feature_store.get_feature_types(schema)

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
                            ColumnTransformer, feature_store=feature_store, prefix=f"{prefix}{column}."
                        ),
                        record_path=feature_metadata.get("record_path"),
                        meta=feature_metadata.get("meta"),
                    )
                    transformers.append((column, transformer, [column]))
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

    def to_onnx(self, name: str | None = None, metrics: list[EvalMetrics] | None = None) -> onnx.ModelProto:
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
        # rename input
        for node in model_onnx.graph.input:
            for feature_name in self.feature_names_in_:
                if node.name == feature_name.replace(".", "_"):
                    node.name = feature_name
        for node in model_onnx.graph.node:
            for i in range(len(node.input)):
                for feature_name in self.feature_names_in_:
                    if node.input[i] == feature_name.replace(".", "_"):
                        node.input[i] = feature_name
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

    def validate(self, model: onnx.ModelProto):
        pass

    def _get_param_names(cls):
        return sklearn.compose.ColumnTransformer._get_param_names()
