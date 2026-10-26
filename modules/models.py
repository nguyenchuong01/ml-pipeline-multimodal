"""Classifier factory: the same names are used in every YAML config."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC


class SubsampleFit(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` on a stratified random subset of the training data.

    Kernel SVMs scale quadratically with the number of samples. Wrapping them
    keeps cross-validation affordable while still evaluating on the full
    validation fold. The subset is drawn inside ``fit`` so there is no leakage.
    """

    def __init__(self, estimator=None, max_samples: int = 10000, random_state: int = 42):
        self.estimator = estimator
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        n = X.shape[0]
        if n > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                k = max(1, int(round(self.max_samples * len(cls_idx) / n)))
                idx.extend(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
            idx = np.sort(np.array(idx))
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def build_classifier(name: str, params: dict[str, Any] | None = None, seed: int = 42):
    """Create a classifier from a short name.

    Supported names: ``logreg``, ``svm_linear``, ``svm_linear_sgd``, ``svm_rbf``, ``random_forest``,
    ``multinomial_nb``, ``complement_nb``.
    """
    params = dict(params or {})
    if name == "logreg":
        defaults = dict(C=1.0, max_iter=2000, class_weight=None, random_state=seed)
        return LogisticRegression(**{**defaults, **params})
    if name == "svm_linear":
        defaults = dict(C=1.0, max_iter=5000, class_weight=None, random_state=seed)
        return LinearSVC(**{**defaults, **params})
    if name == "svm_linear_sgd":
        # Linear SVM (hinge loss) trained with SGD; one-vs-rest is parallelised, so it stays fast with many classes.
        defaults = dict(loss="hinge", alpha=1e-3, max_iter=30, tol=None, n_jobs=-1, class_weight=None, random_state=seed)
        return SGDClassifier(**{**defaults, **params})
    if name == "svm_rbf":
        max_samples = params.pop("max_samples", None)
        defaults = dict(C=1.0, kernel="rbf", gamma="scale", class_weight=None, random_state=seed)
        svc = SVC(**{**defaults, **params})
        return SubsampleFit(svc, max_samples=max_samples, random_state=seed) if max_samples else svc
    if name == "random_forest":
        defaults = dict(n_estimators=300, n_jobs=-1, class_weight=None, random_state=seed)
        return RandomForestClassifier(**{**defaults, **params})
    if name == "multinomial_nb":
        return MultinomialNB(**{**dict(alpha=0.1), **params})
    if name == "complement_nb":
        return ComplementNB(**{**dict(alpha=0.1), **params})
    raise ValueError(f"Unknown classifier '{name}'")
