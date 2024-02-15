import warnings
from itertools import chain
from typing import Literal

import dask.dataframe as dd
import sklearn.compose
from joblib import delayed, Parallel
from sklearn.compose._column_transformer import _get_transformer_list


class ColumnTransformer(sklearn.compose.ColumnTransformer):

    __doc__ = sklearn.compose.ColumnTransformer.__doc__

    def _fit_or_transform(self, X, y=None, func: Literal["fit_transform", "transform"] = None):
        jobs = []
        for _, transformer, columns in self.transformers:
            if isinstance(transformer, str):
                raise NotImplementedError
            _func = getattr(transformer, func)
            jobs.append(delayed(_func)(X[columns]))
        Xs = Parallel(
            n_jobs=len(self.transformers),
            backend="threading",
        )(jobs)

        if self.remainder == "passthrough":
            Xs.append(X[self._passthrough_columns])

        return self._hstack(list(Xs))

    def fit_transform(self, X, y=None, **params):
        self._check_feature_names(X, reset=True)

        # set n_features_in_ attribute
        self._check_n_features(X, reset=True)
        self._validate_transformers()

        self._validate_column_callables(X)
        self._validate_remainder(X)
        _all_transformer_columns = chain(*[columns for _, _, columns in self.transformers])
        self._passthrough_columns = [c for c in X.columns if c not in _all_transformer_columns]

        Xt = self._fit_or_transform(X, y, func="fit_transform")

        self.sparse_output_ = False

        transformers = list(
            self._iter(
                fitted=False,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        )
        self._update_fitted_transformers(transformers)
        # self._validate_output(Xs)
        # self._record_output_indices(Xs)

        return Xt

    def transform(self, X, **params):
        if not hasattr(X, "iloc"):
            raise ValueError("ColumnTransformer only accepts Pandas DataFrames or Dask DataFrames/Series as input")
        return self._fit_or_transform(X, func="transform")

    def _hstack(self, Xs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Concatenating", UserWarning)
            return dd.concat(Xs, axis="columns")


def make_column_transformer(*transformers, **kwargs):
    # This is identical to scikit-learn's. We're just using our
    # ColumnTransformer instead.
    n_jobs = kwargs.pop("n_jobs", 1)
    remainder = kwargs.pop("remainder", "drop")
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(
        transformer_list,
        n_jobs=n_jobs,
        remainder=remainder,
    )


make_column_transformer.__doc__ = sklearn.compose.make_column_transformer.__doc__
