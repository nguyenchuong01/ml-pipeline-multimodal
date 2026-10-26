"""Experiment grid for the text task: feature extraction method x classifier."""
from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from modules.evaluation import run_cv_grid
from modules.models import build_classifier
from modules.text.features import build_sparse_pipeline


def _split_params(params: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    params = dict(params)
    svd = params.pop("svd_components", None)
    return params, svd


def iter_sparse_runs(cfg: dict[str, Any], raw_texts: Sequence[str], tok_texts: Sequence[str], y,
                     features: list[str] | None = None, models: list[str] | None = None) -> Iterator[tuple]:
    """Yield CV runs for TF-IDF / BoW features. Vectorizers are fitted inside each CV fold."""
    seed = cfg.get("seed", 42)
    y = np.asarray(y)
    for fname, fcfg in cfg["features"].items():
        if features and fname not in features:
            continue
        X = np.asarray(tok_texts if fcfg["type"].endswith("_word") else raw_texts, dtype=object)
        for mname, params in cfg["models_sparse"].items():
            if models and mname not in models:
                continue
            params, svd = _split_params(params)
            # Naive Bayes needs non-negative counts/TF-IDF: fine here; RF needs dense -> SVD.
            clf = build_classifier(mname, params, seed=seed)
            yield fname, mname, build_sparse_pipeline(fcfg, clf, svd_components=svd), X, y


def iter_dense_runs(cfg: dict[str, Any], embeddings: dict[str, np.ndarray], y,
                    models: list[str] | None = None) -> Iterator[tuple]:
    """Yield CV runs for dense embedding matrices (already computed, no fitting involved)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seed = cfg.get("seed", 42)
    for ename, X in embeddings.items():
        for mname, params in cfg["models_dense"].items():
            if models and mname not in models:
                continue
            clf = build_classifier(mname, params, seed=seed)
            est = make_pipeline(StandardScaler(), clf) if mname.startswith("svm") or mname == "logreg" else clf
            yield f"emb:{ename}", mname, est, X, np.asarray(y)


def run_text_experiments(cfg: dict[str, Any], runs: Iterator[tuple]) -> pd.DataFrame:
    return run_cv_grid(runs, n_splits=cfg.get("cv_folds", 5), seed=cfg.get("seed", 42))
