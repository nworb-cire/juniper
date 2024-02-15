import logging

import pandas as pd
import pyarrow as pa
from sklearn.base import TransformerMixin, BaseEstimator

from src.common import schema_tools
from src.data_loading.json_normalize import json_normalize


class ColumnNormalizer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        column_name: str,
        schema_in: pa.Schema,
        record_path: str = None,
        meta: list[str | list[str]] = None,
    ):
        super().__init__()
        self.column_name = column_name  # TODO: this can probably be deduced
        self.schema_in = schema_in
        self.record_path = record_path
        self.meta = meta

        if self.record_path is None:
            self.record_prefix = f"{self.column_name}."
        else:
            self.record_prefix = f"{self.column_name}.{self.record_path}."
        self.meta_prefix = f"{self.column_name}."

        schema_out = schema_tools.get_field_schema(schema_in.field(self.column_name))
        # remove fields that will not be in the schema
        _meta = self.meta or []
        for field in schema_out:
            if not field.name.startswith((self.record_prefix, *[f"{self.meta_prefix}{m}" for m in meta or []])):
                logging.debug(f"Removing field {field.name}")
                schema_out = schema_out.remove(schema_out.get_field_index(field.name))
        self.schema_out = schema_out

    def _json_normalize(self, x):
        return json_normalize(
            x,
            record_path=self.record_path,
            meta=self.meta,
            record_prefix=self.record_prefix,
            meta_prefix=self.meta_prefix,
        )

    def fit(self, X, y=None):
        Xt = X[self.column_name].dropna().explode().dropna()
        schema_df = self._json_normalize(Xt.head()).iloc[:0, :]  # empty dataframe for schema
        self.schema = schema_df
        return self

    def transform(self, X) -> pd.DataFrame:
        assert X.shape[1] == 1, "ColumnNormalizer can only handle a single column"
        Xt = X[self.column_name].dropna().explode().dropna()
        if hasattr(Xt, "compute"):
            Xt = Xt.map_partitions(self._json_normalize, meta=self.schema)
        else:
            Xt = self._json_normalize(Xt)
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
