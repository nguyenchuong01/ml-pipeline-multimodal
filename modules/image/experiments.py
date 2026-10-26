"""Classifier grid on top of saved image features."""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.evaluation import run_cv_grid
from modules.image.features import get_features
from modules.models import build_classifier


def make_estimator(model_name: str, params: dict[str, Any], pca: float | int | None = None, seed: int = 42) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    steps.append(("model", build_classifier(model_name, params, seed=seed)))
    return Pipeline(steps)


def compute_all_features(cfg: dict[str, Any], meta: pd.DataFrame, experiments: list[str] | None = None) -> dict[str, np.ndarray]:
    """Extract (or load from cache) the features of every experiment in the config."""
    out = {}
    run = cfg.get("runtime", {})
    for name, exp in cfg["experiments"].items():
        if experiments and name not in experiments:
            continue
        out[name] = get_features(meta, exp["extractor"], out_dir=cfg.get("features_dir", "features/image"),
                                 batch_size=run.get("batch_size", 64), num_workers=run.get("num_workers", 2))
    return out


def iter_runs(cfg: dict[str, Any], feats: dict[str, np.ndarray], meta: pd.DataFrame,
              models: list[str] | None = None) -> Iterator[tuple]:
    """Cross-validation runs on the official training split only."""
    train = meta["is_train"].to_numpy() == 1
    y = meta.loc[train, "class_id"].to_numpy()
    seed = cfg.get("seed", 42)
    only_on = cfg.get("models_only_on", {})  # e.g. random_forest is slow with 200 classes -> run on a subset
    for name, X in feats.items():
        pca = cfg["experiments"][name].get("pca")
        for mname, params in cfg["models"].items():
            if models and mname not in models:
                continue
            if mname in only_on and name not in only_on[mname]:
                continue
            yield name, mname, make_estimator(mname, params, pca, seed), X[train], y


def run_image_experiments(cfg: dict[str, Any], feats, meta, models=None) -> pd.DataFrame:
    return run_cv_grid(iter_runs(cfg, feats, meta, models), n_splits=cfg.get("cv_folds", 3), seed=cfg.get("seed", 42))
