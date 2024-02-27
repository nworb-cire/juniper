import numpy as np
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
        if hasattr(a, "iloc"):
            # dataframe
            if self.dtype == "datetime64[ns]":
                a = a.astype(int)
        return a.astype(self.dtype)

    def fit(self, X, y=None, **fit_params):
        self._cast(X)
        return self

    def transform(self, X, y=None):
        return self._cast(X)
