import warnings
from typing import Union, Any, Optional, List

import dask.dataframe as dd
import sklearn
from dask_ml._typing import ArrayLike, DataFrameType, SeriesType
from dask_ml.preprocessing.data import _handle_zeros_in_scale
from dask_ml.utils import check_array
from sklearn.utils.validation import check_is_fitted


class RobustScaler(sklearn.preprocessing.RobustScaler):

    __doc__ = sklearn.preprocessing.RobustScaler.__doc__

    def _check_array(
        self, X: Union[ArrayLike, DataFrameType], *args: Any, **kwargs: Any
    ) -> Union[ArrayLike, DataFrameType]:
        X = check_array(X, accept_dask_dataframe=True, **kwargs)
        return X

    def fit(
        self,
        X: Union[ArrayLike, DataFrameType],
        y: Optional[Union[ArrayLike, SeriesType]] = None,
    ) -> "RobustScaler":
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" % str(self.quantile_range))

        if not isinstance(X, dd.DataFrame):
            raise NotImplementedError(f"Only Dask DataFrame is supported, got {type(X)}")
        # If the scaler is downstream of another transformer, e.g. an imputer, then all values may not be numerical
        self.boolean_columns = X.select_dtypes(include=bool).columns
        self.non_boolean_columns = X.columns.difference(self.boolean_columns)
        # TODO: Implement https://dl.acm.org/doi/10.1145/375663.375670
        quantiles = X[self.non_boolean_columns].quantile([q_min / 100.0, 0.5, q_max / 100.0]).values.T.compute()

        self.center_: List[float] = quantiles[:, 1]
        self.scale_: List[float] = quantiles[:, 2] - quantiles[:, 0]
        self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        self.n_features_in_: int = X.shape[1]
        return self

    def transform(
        self, X: Union[ArrayLike, DataFrameType]
    ) -> Union[ArrayLike, DataFrameType]:
        """Center and scale the data.

        Can be called on sparse input, provided that ``RobustScaler`` has been
        fitted to dense input and ``with_centering=False``.

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.

        This implementation was copied and modified from Scikit-Learn.

        See License information here:
        https://github.com/scikit-learn/scikit-learn/blob/main/README.rst
        """
        if self.with_centering:
            check_is_fitted(self, "center_")
        if self.with_scaling:
            check_is_fitted(self, "scale_")
        X = self._check_array(X, self.copy)

        # if sparse.issparse(X):
        #     if self.with_scaling:
        #         inplace_column_scale(X, 1.0 / self.scale_)
        # else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Concatenating", UserWarning)
            if self.with_centering:
                Xt = X[self.non_boolean_columns] - self.center_
                X = dd.concat([Xt, X[self.boolean_columns]], axis=1)
            if self.with_scaling:
                Xt = X[self.non_boolean_columns] / self.scale_
                X = dd.concat([Xt, X[self.boolean_columns]], axis=1)
        return X

    def inverse_transform(
        self, X: Union[ArrayLike, DataFrameType]
    ) -> Union[ArrayLike, DataFrameType]:
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.

        This implementation was copied and modified from Scikit-Learn.

        See License information here:
        https://github.com/scikit-learn/scikit-learn/blob/main/README.rst
        """
        check_is_fitted(self, ["center_", "scale_"])

        # if sparse.issparse(X):
        #     if self.with_scaling:
        #         inplace_column_scale(X, self.scale_)
        # else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Concatenating", UserWarning)
            if self.with_scaling:
                Xt = X[self.non_boolean_columns] * self.scale_
                X = dd.concat([Xt, X[self.boolean_columns]], axis=1)
            if self.with_centering:
                Xt = X[self.non_boolean_columns] + self.center_
                X = dd.concat([Xt, X[self.boolean_columns]], axis=1)
        return X
