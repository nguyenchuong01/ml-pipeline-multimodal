"""CUB-200-2011: download, verify and read the official metadata.

Source: Wah et al. (2011), "The Caltech-UCSD Birds-200-2011 Dataset", CaltechDATA.
Limitation to report: CUB images overlap with ImageNet, so ImageNet-pretrained
backbones may have seen some of the test images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from modules.utils import download, safe_extract_tar


def download_cub(url: str, data_dir: str | Path = "data/cub", expected_md5: str | None = None) -> Path:
    """Download (md5-verified, resumable) and extract the archive; returns ``CUB_200_2011``."""
    data_dir = Path(data_dir)
    root = data_dir / "CUB_200_2011"
    if not (root / "images").is_dir():
        archive = download(url, data_dir / "CUB_200_2011.tgz", expected_md5=expected_md5)
        safe_extract_tar(archive, data_dir)
    return root


def load_metadata(root: str | Path) -> pd.DataFrame:
    """One row per image with the official split, class name and bounding box.

    Columns: ``image_id, path, class_id, class_name, is_train, bbox_x, bbox_y, bbox_w, bbox_h``.
    The rows are sorted by ``image_id`` so that row ``i`` always matches feature row ``i``.
    """
    root = Path(root)
    images = pd.read_csv(root / "images.txt", sep=" ", header=None, names=["image_id", "rel_path"])
    labels = pd.read_csv(root / "image_class_labels.txt", sep=" ", header=None, names=["image_id", "class_id"])
    split = pd.read_csv(root / "train_test_split.txt", sep=" ", header=None, names=["image_id", "is_train"])
    boxes = pd.read_csv(root / "bounding_boxes.txt", sep=" ", header=None,
                        names=["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    classes = pd.read_csv(root / "classes.txt", sep=" ", header=None, names=["class_id", "class_name"])
    df = images.merge(labels, on="image_id").merge(split, on="image_id").merge(boxes, on="image_id")
    df = df.merge(classes, on="class_id")
    df["path"] = df["rel_path"].map(lambda p: str(root / "images" / p))
    df["class_name"] = df["class_name"].str.replace(r"^\d+\.", "", regex=True).str.replace("_", " ")
    df["class_id"] = df["class_id"] - 1  # 0..199
    return df.sort_values("image_id").reset_index(drop=True)[
        ["image_id", "path", "class_id", "class_name", "is_train", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ]


def prepare_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    d = cfg["data"]
    root = download_cub(d["url"], d.get("dir", "data/cub"), d.get("md5"))
    return load_metadata(root)
