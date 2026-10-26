"""OkCupid (JSE) data loading, cleaning and target construction.

Source: Kim & Escobedo-Land (2015), "OkCupid Profile Data for Introductory
Statistics and Data Science Courses", Journal of Statistics Education.
The file is downloaded from the authors' public GitHub repository.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.utils import download, safe_extract_zip

EDUCATION_LEVELS = ["high school", "two-year college", "college/university", "masters", "advanced degree"]
STATUS_PREFIXES = ("graduated from", "working on", "dropped out of")


def load_raw(url: str, data_dir: str | Path = "data/okcupid") -> pd.DataFrame:
    """Download (if needed) and read ``profiles_revised.csv``."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "profiles_revised.csv"
    if not csv_path.exists():
        zip_path = download(url, data_dir / "profiles_revised.csv.zip")
        safe_extract_zip(zip_path, data_dir)
    return pd.read_csv(csv_path)


def _clean_text(value: Any) -> Any:
    if isinstance(value, str):
        return html.unescape(value).strip().lower()
    return value


def _education_level(value: Any) -> Any:
    """Collapse the 32 raw education values into 5 ordered levels (NaN if unknown)."""
    if not isinstance(value, str):
        return np.nan
    v = value.lower()
    if "ph.d" in v or "law school" in v or "med school" in v:
        return "advanced degree"
    if "masters" in v:
        return "masters"
    if "college/university" in v:
        return "college/university"
    if "two-year" in v:
        return "two-year college"
    if "high school" in v:
        return "high school"
    return np.nan  # 'space camp' and similar are not a level


def _education_status(value: Any) -> Any:
    if not isinstance(value, str):
        return np.nan
    for prefix in STATUS_PREFIXES:
        if value.lower().startswith(prefix):
            return prefix
    return np.nan


def _base_label(value: Any, separators: tuple[str, ...]) -> Any:
    """Keep the leading word(s) before a modifier such as 'but not too serious about it'."""
    if not isinstance(value, str):
        return np.nan
    for sep in separators:
        value = value.split(sep)[0]
    return value.strip()


def clean_profiles(df: pd.DataFrame, income_missing_value: int | None = -1, height_range=(48, 96)) -> pd.DataFrame:
    """Clean raw profiles.

    * decode HTML entities (``&rsquo;``) and lower-case strings,
    * turn the disguised missing value ``income == -1`` into NaN,
    * set implausible heights (<4 ft or >8 ft) to NaN,
    * derive ``education_level`` / ``education_status`` / ``religion_base`` / ``sign_base``,
    * strip language proficiency from ``speaks``.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_clean_text)

    if income_missing_value is not None and "income" in out:
        out["income"] = out["income"].replace(income_missing_value, np.nan)
    if "height" in out:
        lo, hi = height_range
        out.loc[(out["height"] < lo) | (out["height"] > hi), "height"] = np.nan

    out["education_level"] = out["education"].map(_education_level)
    out["education_status"] = out["education"].map(_education_status)
    out["religion_base"] = out["religion"].map(lambda v: _base_label(v, (" and ", " but ")))
    out["sign_base"] = out["sign"].map(lambda v: _base_label(v, (" and ", " but ")))
    out["speaks_langs"] = out["speaks"].map(
        lambda v: ", ".join(re.sub(r"\(.*?\)", "", p).strip() for p in v.split(",")) if isinstance(v, str) else np.nan
    )
    return out


def make_target(df: pd.DataFrame, stem_jobs: list[str], exclude_jobs: list[str], target_col: str = "is_stem") -> pd.DataFrame:
    """Binary target: 1 if the declared job field is STEM, 0 for the other known fields.

    Rows with a missing job or with a non-informative job label (``other``,
    ``student``, ``retired`` ...) are removed. ``job`` itself is dropped from the
    features afterwards because the target is derived from it (label leakage).
    """
    keep = df["job"].notna() & ~df["job"].isin(exclude_jobs)
    out = df.loc[keep].copy()
    out[target_col] = out["job"].isin(stem_jobs).astype(int)
    return out.reset_index(drop=True)


def prepare_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Full data step: download, clean, build target, return ``(X, y)``."""
    d = cfg["data"]
    raw = load_raw(d["url"], d.get("dir", "data/okcupid"))
    clean = clean_profiles(raw, d.get("income_missing_value", -1))
    labeled = make_target(clean, d["stem_jobs"], d["exclude_jobs"], d.get("target", "is_stem"))
    y = labeled[d.get("target", "is_stem")]
    drop = set(d.get("drop_columns", [])) | {d.get("target", "is_stem"), "job"}
    X = labeled.drop(columns=[c for c in drop if c in labeled.columns])
    return X, y
