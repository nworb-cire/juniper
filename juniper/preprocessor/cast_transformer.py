import numpy as np
import pandas as pd
from skl2onnx import update_registered_converter
from skl2onnx.operator_converters.cast_op import convert_sklearn_cast
from skl2onnx.shape_calculators.cast_op import calculate_sklearn_cast_transformer
from sklearn.base import TransformerMixin, BaseEstimator


class CastTransformer(TransformerMixin, BaseEstimator):
    """
    Taken from skl2onnx.sklapi.CastTransformer

    Cast features into a specific types.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double.

    Parameters
    ----------
    dtype : numpy type,
        output are cast into that type
    """

    def __init__(self, *, dtype=np.float32):
        self.dtype = dtype

    def set_output(self, *, transform=None):
        return self

    def _cast(self, a):
        try:
            return a.astype(self.dtype)
        except (TypeError, ValueError) as e:
            err_str = f"Unable to cast to {self.dtype}: {e}"
            if hasattr(a, "columns"):
                err_str += f"\nColumns: {a.columns}"
            raise TypeError(err_str) from e

    def fit(self, X, y=None, **fit_params):
        # self._cast(X)
        return self

    def transform(self, X, y=None):
        return self._cast(X)


update_registered_converter(
    CastTransformer,
    "JuniperCastTransformer",
    calculate_sklearn_cast_transformer,
    convert_sklearn_cast,
)


class DatetimeCastTransformer(CastTransformer):
    def __init__(self, fillna: pd.Timestamp | None = None):
        if fillna is None:
            raise NotImplementedError("fillna must be provided for now.")
        self.fillna = fillna
        super().__init__(dtype=np.int64)

    def _cast(self, a):
        zero = pd.Timestamp("0001-01-01T00:00:00Z")
        a[a == zero] = self.fillna
        for col in a.columns:
            try:
                # TODO: Handle nan with a nullable integer type?
                a[col] = pd.to_datetime(a[col], utc=True).apply(
                    lambda x: self.fillna.timestamp() if pd.isna(x) else x.timestamp()
                )
            except (TypeError, AttributeError) as e:
                raise TypeError(f"Column {col} is not a datetime") from e
        return super()._cast(a)


# TODO: Validate
update_registered_converter(
    DatetimeCastTransformer,
    "JuniperDatetimeCastTransformer",
    calculate_sklearn_cast_transformer,
    convert_sklearn_cast,
)
