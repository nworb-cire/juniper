import logging
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import TransformerMixin, BaseEstimator

from juniper.common import schema_tools
from juniper.common.data_type import FeatureType
from juniper.common.setup import load_config
from juniper.data_loading.json_normalize import json_normalize


class ColumnNormalizer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        field: pa.Field,
        preprocessor_factory: Callable,
        record_path: str = None,
        meta: list[str | list[str]] = None,
    ):
        self.field = field
        self.preprocessor_factory = preprocessor_factory
        self.record_path = record_path
        self.meta = meta

        config = load_config()
        override_unusable_features = tuple(config["data_sources"]["feature_store"].get("unusable_features", []))

        if self.record_path is None:
            self.record_prefix = f"{self.field.name}."
        else:
            self.record_prefix = f"{self.field.name}.{self.record_path}."
        self.meta_prefix = f"{self.field.name}."

        schema_out = schema_tools.get_field_schema(field)
        # remove fields that will not be in the schema
        remove_list = []
        for field in schema_out:
            if (
                not field.name.startswith((self.record_prefix, *[f"{self.meta_prefix}{m}" for m in meta or []]))
                or field.metadata.get(b"usable_type", b"").decode() == FeatureType.UNUSABLE
                or field.name.startswith(override_unusable_features)
            ):
                logging.debug(f"Removing field {field.name}")
                remove_list.append(field.name)
        schema_out = pa.schema([field for field in schema_out if field.name not in remove_list])
        self.schema_out = schema_out

        if (
            all(field.metadata[b"usable_type"].decode() == FeatureType.UNUSABLE for field in schema_out)
            or len(schema_out) == 0
        ):
            logging.warning(f"Array column {self.field.name} is unusable and will be dropped")
            self.column_transformer = None
            return

        self.column_transformer = preprocessor_factory(self.schema_out)

    def set_output(self, *, transform=None):
        return self

    def _sk_visual_block_(self):
        return self.column_transformer._sk_visual_block_()

    def _json_normalize(self, x):
        return json_normalize(
            x,
            record_path=self.record_path,
            meta=self.meta,
            record_prefix=self.record_prefix if self.record_path is not None else None,
            meta_prefix=self.meta_prefix if self.meta is not None else None,
        )

    def _transform(self, X):
        assert X.shape[1] == 1, "ColumnNormalizer can only handle a single column"
        Xt = X[self.field.name].explode().dropna()
        Xt = self._json_normalize(Xt)
        Xt = Xt.rename(columns={c: f"{self.field.name}.{c}" for c in Xt.columns if not c.startswith(self.field.name)})
        return Xt

    def _flatten(self, X, idx: pd.Index):
        Xt = X.groupby(X.index).agg(lambda x: x.tolist())
        Xt = Xt.apply(lambda row: np.array([*row]), axis=1)
        Xt = Xt.reindex(idx)
        Xt = Xt.apply(lambda x: x if hasattr(x, "__len__") else np.array([[]]))  # fixme
        Xt = Xt.to_frame(name=self.field.name)
        return Xt

    def fit_transform(self, X, y=None, **fit_params: dict):
        Xt = self._transform(X)
        index = Xt.index
        for _, _, columns in self.column_transformer.transformers:
            for column in columns:
                if column not in Xt.columns:
                    raise ValueError(
                        f"Column {column} not found in input data {Xt.columns} for field {self.field.name} "
                        + "(hint: either check the record path or add it to the remove list in the config file)"
                    )
        Xt = self.column_transformer.fit_transform(Xt, y, **fit_params)
        Xt = pd.DataFrame(Xt, index=index)
        Xt = self._flatten(Xt, X.index)
        return Xt

    def fit(self, X, y=None, **fit_params: dict):
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X) -> pd.DataFrame:
        Xt = self._transform(X)
        index = Xt.index
        if Xt.empty:
            # This should only be the case when every value in X was None, TODO verify
            n = X.shape[0]
            Xt = pd.DataFrame({field.name: [None] * n for field in self.schema_out}, index=X.index)
        Xt = self.column_transformer.transform(Xt)
        Xt = pd.DataFrame(Xt, index=index)
        Xt = self._flatten(Xt, X.index)
        return Xt
