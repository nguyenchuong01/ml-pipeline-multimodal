"""Configurable preprocessing for mixed-type tabular data.

Every transformer is fitted **inside** the cross-validation pipeline, i.e. only
on the training fold, so no statistic of the validation fold leaks into training.

``build_preprocessor`` reads a ``preprocess`` dict (see ``configs/tabular.yaml``):

    missing:            simple | indicator | drop_high
    nominal_encoding:   onehot | frequency | ordinal
    scaler:             none | standard | minmax
    pca:                null | float in (0,1) (explained variance) | int
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


class ColumnsIn:
    """Callable column selector: keep ``cols`` that are still present in X (picklable)."""

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)

    def __call__(self, X: pd.DataFrame) -> list[str]:
        return [c for c in self.cols if c in X.columns]


class DropHighMissing(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate on the training fold exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        rate = X.isna().mean()
        self.dropped_ = rate[rate > self.threshold].index.tolist()
        self.keep_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_, dtype=object)


class OrderedEncoder(BaseEstimator, TransformerMixin):
    """Map ordered categories to ranks 0..k-1 by column *name* (unknown / missing -> NaN).

    Working by name keeps it correct when ``DropHighMissing`` removes some columns.
    """

    def __init__(self, orders: dict | None = None):
        self.orders = orders

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.full(X.shape, np.nan, dtype=float)
        for j, c in enumerate(X.columns):
            ranks = {v: i for i, v in enumerate(self.orders[c])}
            out[:, j] = X[c].map(ranks).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_rank" for c in self.columns_], dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency in the training data (unseen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        out = np.zeros(X.shape, dtype=float)
        for j, c in enumerate(self.columns_):
            out[:, j] = X[c].map(self.maps_[c]).fillna(0.0).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_], dtype=object)


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encoding for comma-separated fields such as ``ethnicity`` or ``speaks``.

    Tokens seen fewer than ``min_count`` times in the training fold are ignored.
    Missing values produce an all-zero row.
    """

    def __init__(self, min_count: int = 50, sep: str = ","):
        self.min_count = min_count
        self.sep = sep

    def _tokens(self, value) -> list[str]:
        if not isinstance(value, str):
            return []
        return [t.strip() for t in value.split(self.sep) if t.strip()]

    def fit(self, X, y=None):
        col = pd.Series(np.asarray(X).ravel())
        counts: dict[str, int] = {}
        for v in col:
            for t in set(self._tokens(v)):
                counts[t] = counts.get(t, 0) + 1
        self.vocab_ = sorted(t for t, n in counts.items() if n >= self.min_count)
        self.index_ = {t: i for i, t in enumerate(self.vocab_)}
        return self

    def transform(self, X):
        col = pd.Series(np.asarray(X).ravel())
        out = np.zeros((len(col), len(self.vocab_)), dtype=float)
        for i, v in enumerate(col):
            for t in self._tokens(v):
                j = self.index_.get(t)
                if j is not None:
                    out[i, j] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features is not None and len(input_features) else "multi"
        return np.array([f"{prefix}={t}" for t in self.vocab_], dtype=object)


def _scaler(cfg: dict[str, Any]):
    kind = cfg.get("scaler", "standard")
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        lo, hi = cfg.get("minmax_range", [0, 1])
        return MinMaxScaler(feature_range=(lo, hi))
    if kind in (None, "none"):
        return "passthrough"
    raise ValueError(f"Unknown scaler '{kind}'")


def build_preprocessor(cfg: dict[str, Any]) -> Pipeline:
    """Build an unfitted sklearn ``Pipeline`` implementing the preprocess config."""
    missing = cfg.get("missing", "indicator")
    nominal_encoding = cfg.get("nominal_encoding", "onehot")
    numeric = list(cfg.get("numeric", []))
    ordinal = cfg.get("ordinal", {}) or {}
    nominal = list(cfg.get("nominal", []))
    multilabel = list(cfg.get("multilabel", []))
    min_freq = cfg.get("min_category_count", 30)
    use_indicator = missing == "indicator"

    # numeric: impute (+ optional missing flag) -> scale
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=use_indicator)), ("scale", _scaler(cfg))]

    # ordinal: map to ordered integers (NaN stays NaN) -> impute -> scale
    ord_steps = [
        ("encode", OrderedEncoder(orders=ordinal)),
        ("impute", SimpleImputer(strategy="median" if use_indicator else "most_frequent", add_indicator=use_indicator)),
        ("scale", _scaler(cfg)),
    ]

    # nominal: impute (a constant 'missing' category acts as an indicator) -> encode
    nom_impute = (SimpleImputer(strategy="constant", fill_value="missing") if use_indicator
                  else SimpleImputer(strategy="most_frequent"))
    if nominal_encoding == "onehot":
        nom_encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq, sparse_output=False)
        nom_steps = [("impute", nom_impute), ("encode", nom_encoder)]
    elif nominal_encoding == "frequency":
        nom_steps = [("impute", nom_impute), ("encode", FrequencyEncoder()), ("scale", _scaler(cfg))]
    elif nominal_encoding == "ordinal":  # arbitrary integer codes: a deliberately weak baseline
        nom_steps = [("impute", nom_impute),
                     ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                     ("scale", _scaler(cfg))]
    else:
        raise ValueError(f"Unknown nominal_encoding '{nominal_encoding}'")

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), ColumnsIn(numeric)))
    if ordinal:
        transformers.append(("ord", Pipeline(ord_steps), ColumnsIn(list(ordinal))))
    if nominal:
        transformers.append(("nom", Pipeline(nom_steps), ColumnsIn(nominal)))
    for col in multilabel:
        transformers.append((f"multi_{col}", MultiHotEncoder(min_count=cfg.get("multihot_min_count", 100)), ColumnsIn([col])))

    # MultiHotEncoder receives a one-column frame and flattens it with np.asarray(...).ravel().
    column_transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    steps: list[tuple[str, Any]] = []
    if missing == "drop_high":
        steps.append(("drop_high_missing", DropHighMissing(cfg.get("high_missing_threshold", 0.5))))
    steps.append(("columns", column_transformer))
    pca = cfg.get("pca")
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=cfg.get("seed", 42))))
    return Pipeline(steps)
