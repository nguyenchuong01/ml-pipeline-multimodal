"""Plot helpers (matplotlib/seaborn). All functions return the figure and optionally save it."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def _finish(fig, save_path: str | Path | None):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_missing(df: pd.DataFrame, top: int = 20, save_path=None):
    rate = df.isna().mean().sort_values(ascending=False).head(top) * 100
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rate) + 1.5))
    sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#4C78A8")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing rate per column")
    return _finish(fig, save_path)


def plot_class_distribution(y: Sequence, title: str = "Class distribution", save_path=None, horizontal: bool = True):
    counts = pd.Series(y).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(counts) + 1.5) if horizontal else (8, 4))
    if horizontal:
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="#59A14F")
        ax.set_xlabel("Samples")
    else:
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#59A14F")
        ax.set_ylabel("Samples")
    ax.set_title(title)
    return _finish(fig, save_path)


def plot_confusion(cm: np.ndarray, labels: Sequence[str], normalize: bool = True, title: str = "Confusion matrix", save_path=None):
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    size = max(5, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(mat, annot=len(labels) <= 15, fmt=".2f" if normalize else ".0f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _finish(fig, save_path)


def plot_results(df: pd.DataFrame, x: str = "experiment", hue: str = "model", metric: str = "macro_f1",
                 title: str | None = None, save_path=None):
    """Grouped bar chart of ``metric`` mean with std error bars."""
    data = df[df["status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * data[x].nunique() + 3), 4.5))
    experiments = list(dict.fromkeys(data[x]))
    models = list(dict.fromkeys(data[hue]))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = data[data[hue] == m].set_index(x).reindex(experiments)
        ax.bar(np.arange(len(experiments)) + i * width, sub[f"{metric}_mean"], width,
               yerr=sub[f"{metric}_std"], label=m, capsize=2)
    ax.set_xticks(np.arange(len(experiments)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(experiments, rotation=35, ha="right")
    low = (data[f"{metric}_mean"] - data[f"{metric}_std"]).min()
    high = (data[f"{metric}_mean"] + data[f"{metric}_std"]).max()
    ax.set_ylim(max(0.0, low - 0.05), min(1.0, high + 0.03))  # zoomed axis: differences are small
    ax.set_ylabel(metric.replace("_", " ") + " (y-axis truncated)")
    ax.set_title(title or f"{metric} by experiment")
    ax.legend(loc="lower right", fontsize=8)
    return _finish(fig, save_path)
