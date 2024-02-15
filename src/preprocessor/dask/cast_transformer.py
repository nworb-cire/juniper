# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class CastTransformer(TransformerMixin, BaseEstimator):
    """
    Cast features into a specific types.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double.

    Parameters
    ----------
    dtype : numpy type,
        output are cast into that type
    """  # noqa

    def __init__(self, *, dtype=np.float32):
        self.dtype = dtype

    def _cast(self, a, name):
        try:
            a2 = a.astype(self.dtype)
        except ValueError as e:
            raise ValueError("Unable to cast {} from {} into {}.".format(name, a.dtype, self.dtype)) from e
        return a2

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, y=None):
        """
        Casts array X.
        """
        return self._cast(X, "X")
