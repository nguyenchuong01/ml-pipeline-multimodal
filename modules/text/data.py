"""livedoor News Corpus: download, parse and split.

Source: RONDHUIT Co., Ltd. https://www.rondhuit.com/download.html
License: Creative Commons BY-ND 2.1 JP (attribution requirements differ by
category, see ``LICENSE.txt`` in each category folder). Raw text must not be
redistributed or committed to the repository.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from modules.utils import download, safe_extract_tar


def download_corpus(url: str, data_dir: str | Path = "data/livedoor") -> Path:
    """Download and extract the corpus; returns the ``text`` folder."""
    data_dir = Path(data_dir)
    text_dir = data_dir / "text"
    if not text_dir.exists():
        archive = download(url, data_dir / "ldcc.tgz")
        safe_extract_tar(archive, data_dir)
    return text_dir


def load_corpus(text_dir: str | Path) -> pd.DataFrame:
    """Parse every article into one row: ``category, article_id, url, date, title, body``.

    File layout: line 1 = URL, line 2 = date, line 3 = title, remaining lines = body.
    """
    text_dir = Path(text_dir)
    rows = []
    for cat_dir in sorted(p for p in text_dir.iterdir() if p.is_dir()):
        for path in sorted(cat_dir.glob(f"{cat_dir.name}-*.txt")):
            lines = path.read_text(encoding="utf-8").split("\n")
            rows.append({
                "category": cat_dir.name,
                "article_id": path.stem,
                "url": lines[0].strip(),
                "date": lines[1].strip(),
                "title": lines[2].strip(),
                "body": "\n".join(lines[3:]).strip(),
            })
    return pd.DataFrame(rows)


def build_text(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Concatenate the selected fields (``title``, ``body``) into the model input."""
    return df[fields].fillna("").agg("\n".join, axis=1)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (the corpus has no official split)."""
    train, test = train_test_split(df, test_size=test_size, stratify=df["category"], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def duplicate_report(df: pd.DataFrame, fields: list[str]) -> dict[str, int]:
    """Count exact duplicate texts (a quick leakage check between train and test)."""
    text = build_text(df, fields)
    return {"n_articles": len(df), "n_duplicate_texts": int(text.duplicated().sum()),
            "n_duplicate_titles": int(df["title"].duplicated().sum())}


def prepare_dataset(cfg: dict[str, Any]):
    """Download, parse, split. Returns ``(df_all, train_df, test_df)``."""
    d = cfg["data"]
    df = load_corpus(download_corpus(d["url"], d.get("dir", "data/livedoor")))
    train, test = split_dataset(df, d.get("test_size", 0.2), cfg.get("seed", 42))
    return df, train, test
