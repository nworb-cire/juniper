import warnings

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


from dask import dataframe as dd

class MissingIndicator(TransformerMixin, BaseEstimator):
    def __init__(self, *, missing_values=np.nan):
        self.missing_values = missing_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.isna()
        df.columns = [f"{col}_missing" for col in df.columns]
        return df


class ConstantImputer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        missing_values=np.nan,
        fill_value=0.0,
        add_indicator=False,
    ):
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        if add_indicator:
            self.indicator_ = MissingIndicator(missing_values=missing_values)
        else:
            self.indicator_ = None

    def _validate_input(self, X):
        if not hasattr(X, "iloc"):
            raise TypeError("X must be a dataframe.")
        
    def _fit_indicator(self, X):
        if self.indicator_ is not None:
            self.indicator_.fit(X)
        return self

    def fit(self, X, y=None):
        self._validate_input(X)
        self._fit_indicator(X)
        return self

    def transform(self, X):
        self._validate_input(X)
        Xt = X.fillna(self.fill_value)
        if self.indicator_ is not None:
            mask = self.indicator_.transform(X)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Concatenating", UserWarning)
                return dd.concat([Xt, mask], axis=1)
        return Xt
