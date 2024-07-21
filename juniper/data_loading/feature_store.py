from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import dataset as ds

from juniper.common.data_type import FeatureType
from juniper.data_loading.data_source import (
    BaseDataSource,
    ParquetDataSource,
)


class BaseFeatureStore(BaseDataSource, ABC):
    def get_metadata(self):
        self.schema = self.get_schema()
        self.feature_types = self.get_feature_types(self.schema)
        return self.feature_types

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        pass

    @classmethod
    @abstractmethod
    def get_feature_types(cls, schema: pa.Schema) -> dict[FeatureType, list[str]]:
        pass


class BaseParquetFeatureStore(BaseFeatureStore, ParquetDataSource, ABC):
    def __init__(
        self,
        /,
        path: Path,
        **kwargs,
    ):
        super().__init__(path=path, **kwargs)

    def _load_train_test(
        self,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        train_time_end: pd.Timestamp | None = None,
        holdout_time_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        filters = ~ds.field(self.index_column).is_null()
        if train_idx is not None:
            filters &= pc.is_in(ds.field(self.index_column), pa.array(train_idx))
        train = self.read_parquet(filters=filters)
        train = train.sort_values(self.timestamp_column)
        train = train[~train.index.duplicated(keep="last")]
        if test_idx is not None:
            filters = ~ds.field(self.index_column).is_null() & pc.is_in(ds.field(self.index_column), pa.array(test_idx))
            test = self.read_parquet(filters=filters)
            test = test.sort_values(self.timestamp_column)
            test = test[~test.index.duplicated(keep="last")]
        else:
            test = None
        return train, test


class ParquetFeatureStore(BaseParquetFeatureStore, ABC):
    def __init__(
        self,
        /,
        **kwargs,
    ):
        enabled_feature_types = kwargs.get("enabled_feature_types", None)
        override_unusable_features = kwargs.get("override_unusable_features", None)
        super().__init__(**kwargs)
        self.enabled_feature_types = enabled_feature_types or list(FeatureType)
        self.override_unusable_features = override_unusable_features or []
