import logging
from typing import Callable

import pandas as pd
import pyarrow as pa
from sklearn.base import TransformerMixin, BaseEstimator

from juniper.common import schema_tools
from juniper.common.data_type import FeatureType
from juniper.data_loading.feature_store import ParquetFeatureStore
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

        if self.record_path is None:
            self.record_prefix = f"{self.field.name}."
        else:
            self.record_prefix = f"{self.field.name}.{self.record_path}."
        self.meta_prefix = f"{self.field.name}."

        schema_out = schema_tools.get_field_schema(field)
        # remove fields that will not be in the schema
        _meta = self.meta or []
        for field in schema_out:
            if (
                not field.name.startswith((self.record_prefix, *[f"{self.meta_prefix}{m}" for m in meta or []]))
                or field.metadata.get(b"usable_type", b"").decode() == FeatureType.UNUSABLE
            ):
                logging.warning(f"Removing field {field.name}")
                schema_out = schema_out.remove(schema_out.get_field_index(field.name))
        self.schema_out = schema_out

        if (
            all(field.metadata[b"usable_type"].decode() == FeatureType.UNUSABLE for field in schema_out)
            or len(schema_out) == 0
        ):
            logging.warning(f"Array column {self.field.name} is unusable and will be dropped")
            self.column_transfomer = None
            return

        self.column_transfomer = preprocessor_factory(
            ParquetFeatureStore, self.schema_out
        )  # FIXME: this should be an argument

    def set_output(self, *, transform=None):
        return self

    def _json_normalize(self, x):
        return json_normalize(
            x,
            # FIXME
            # record_path=self.record_path,
            # meta=self.meta,
            # record_prefix=self.record_prefix,
            # meta_prefix=self.meta_prefix,
        )

    def _transform(self, X):
        assert X.shape[1] == 1, "ColumnNormalizer can only handle a single column"
        Xt = X[self.field.name].explode()
        Xt = self._json_normalize(Xt)
        Xt.columns = [f"{self.field.name}.{c}" for c in Xt.columns if not c.startswith(self.field.name)]
        # set dtypes
        for field in self.schema_out:
            match field.metadata.get(b"usable_type"):
                case b"unusable":
                    raise NotImplementedError
                case b"numeric":
                    Xt[field.name] = Xt[field.name].astype("float32")
                case _:
                    pass
        return Xt

    def _flatten(self, Xt):
        return Xt.groupby(Xt.index).agg(lambda x: x.tolist())

    def fit_transform(self, X, y=None, **fit_params: dict):
        Xt = self._transform(X)
        index = Xt.index
        Xt = self.column_transfomer.fit_transform(Xt, y, **fit_params)
        Xt = pd.DataFrame(Xt, index=index)
        Xt = self._flatten(Xt)
        return Xt

    def fit(self, X, y=None, **fit_params: dict):
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X) -> pd.DataFrame:
        Xt = self._transform(X)
        index = Xt.index
        Xt = self.column_transfomer.transform(Xt)
        Xt = pd.DataFrame(Xt, index=index)
        Xt = self._flatten(Xt)
        return Xt
