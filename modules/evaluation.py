"""Cross-validation, hold-out evaluation and result tables shared by all tasks."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy"}


def cross_validate_model(
    estimator,
    X,
    y,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int | None = None,
) -> dict[str, float]:
    """Stratified K-fold CV returning mean and std of macro-F1 and accuracy, plus timing."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(clone(estimator), X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs, error_score="raise")
    return {
        "macro_f1_mean": float(np.mean(res["test_macro_f1"])),
        "macro_f1_std": float(np.std(res["test_macro_f1"])),
        "accuracy_mean": float(np.mean(res["test_accuracy"])),
        "accuracy_std": float(np.std(res["test_accuracy"])),
        "fit_time_s": float(np.mean(res["fit_time"])),
        "predict_time_s": float(np.mean(res["score_time"])),
    }


def run_cv_grid(
    runs: Iterable[tuple[str, str, Any, Any, Any]],
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate many ``(experiment, model_name, estimator, X, y)`` tuples with CV.

    Returns one tidy row per run. Failures are recorded (not raised) so one bad
    configuration does not abort a long ``Run all``.
    """
    rows = []
    for exp, model_name, estimator, X, y in runs:
        start = time.perf_counter()
        try:
            metrics = cross_validate_model(estimator, X, y, n_splits=n_splits, seed=seed, n_jobs=n_jobs)
            row = {"experiment": exp, "model": model_name, **metrics, "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            row = {"experiment": exp, "model": model_name, "status": f"failed: {type(exc).__name__}: {exc}"}
        rows.append(row)
        if verbose:
            elapsed = time.perf_counter() - start
            f1 = row.get("macro_f1_mean", float("nan"))
            print(f"[{exp:<34}] {model_name:<14} macro-F1={f1:.4f}  ({elapsed:.1f}s)  {row['status'][:60]}")
    return pd.DataFrame(rows)


def format_mean_std(df: pd.DataFrame, metric: str = "macro_f1", digits: int = 4) -> pd.Series:
    """Return a 'mean ± std' string column for display in reports."""
    return df[f"{metric}_mean"].round(digits).astype(str) + " ± " + df[f"{metric}_std"].round(digits).astype(str)


def holdout_evaluate(estimator, X_train, y_train, X_test, y_test, labels: list | None = None) -> dict[str, Any]:
    """Fit on the training split, evaluate on the held-out test split."""
    est = clone(estimator)
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = est.predict(X_test)
    predict_time = time.perf_counter() - t0
    return {
        "estimator": est,
        "y_pred": y_pred,
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "confusion": confusion_matrix(y_test, y_pred, labels=labels),
        "report": pd.DataFrame(classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)).T,
    }


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
