import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


PERIODS = {
    "year": 365.2425 * 24 * 60 * 60,
    "week": 7 * 24 * 60 * 60,
    "day": 24 * 60 * 60,
}


class PeriodicTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, periods: dict[str, float] = PERIODS, keep_original: bool = False):
        self.periods = periods
        self.transform_ = "pandas"
        self.keep_original = keep_original

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        out_sin = np.concatenate([np.sin(2 * np.pi * X.values / period) for period in self.periods.values()], axis=1)
        out_cos = np.concatenate([np.cos(2 * np.pi * X.values / period) for period in self.periods.values()], axis=1)
        if self.transform_ == "pandas":
            out_sin = pd.DataFrame(
                out_sin,
                columns=[f"{col}.{p}.sin" for p in self.periods for col in X.columns],
                index=X.index,
            )
            out_cos = pd.DataFrame(
                out_cos,
                columns=[f"{col}.{p}.cos" for p in self.periods for col in X.columns],
                index=X.index,
            )
            if self.keep_original:
                vals = [X]
            else:
                vals = []
            vals.extend([out_sin, out_cos])
            return pd.concat(vals, axis=1)
        if self.keep_original:
            vals = [X.values]
        else:
            vals = []
        vals.extend([out_sin, out_cos])
        return np.concatenate(vals, axis=1)

    def set_output(self, *, transform=None):
        if transform not in (None, "pandas"):
            raise ValueError(f"Invalid transform: {transform}")
        self.transform_ = transform


# TODO: ONNX export
