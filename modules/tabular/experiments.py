"""Run the preprocessing x model grid defined in ``configs/tabular.yaml``."""
from __future__ import annotations

from typing import Any, Iterator

import pandas as pd
from sklearn.pipeline import Pipeline

from modules.config import deep_update
from modules.evaluation import run_cv_grid
from modules.models import build_classifier
from modules.tabular.preprocessing import build_preprocessor


def make_pipeline(preprocess_cfg: dict[str, Any], model_name: str, model_params: dict[str, Any], seed: int = 42) -> Pipeline:
    """Preprocessing + classifier in ONE pipeline so CV fits both on the training fold only."""
    pre = build_preprocessor({**preprocess_cfg, "seed": seed})
    clf = build_classifier(model_name, model_params, seed=seed)
    return Pipeline([("preprocess", pre), ("model", clf)])


def iter_runs(cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series,
              experiments: list[str] | None = None, models: list[str] | None = None) -> Iterator[tuple]:
    for exp_name, overrides in cfg["experiments"].items():
        if experiments and exp_name not in experiments:
            continue
        pre_cfg = deep_update(cfg["preprocess"], overrides)
        for model_name, params in cfg["models"].items():
            if models and model_name not in models:
                continue
            yield exp_name, model_name, make_pipeline(pre_cfg, model_name, params, cfg.get("seed", 42)), X, y


def run_experiments(cfg: dict[str, Any], X: pd.DataFrame, y: pd.Series,
                    experiments: list[str] | None = None, models: list[str] | None = None) -> pd.DataFrame:
    """Cross-validate the selected experiments x models and return the result table."""
    return run_cv_grid(
        iter_runs(cfg, X, y, experiments, models),
        n_splits=cfg.get("cv_folds", 5),
        seed=cfg.get("seed", 42),
        n_jobs=None,  # model-level parallelism (random forest) is already enabled
    )
