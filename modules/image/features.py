"""Image feature extraction.

* ``extract_deep_features``: pretrained CNN / ViT from ``timm`` used as a frozen
  feature extractor (global-pooled embedding). Resize, crop, bounding-box
  cropping and the backbone are all configurable.
* ``extract_hog_features``: traditional HOG descriptor baseline.

Features are saved as ``.npy`` (row ``i`` = row ``i`` of the metadata table).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def feature_filename(spec: dict[str, Any]) -> str:
    """Deterministic file name for an extractor spec, e.g. ``resnet50__r256_c224__bbox.npy``."""
    if spec.get("backbone") == "hog":
        return f"hog__s{spec.get('size', 128)}.npy"
    name = spec["backbone"].replace("/", "-")
    crop_mode = "bbox" if spec.get("use_bbox") else "full"
    return f"{name}__r{spec.get('resize', 256)}_c{spec.get('crop', 224)}__{crop_mode}.npy"


def _crop_bbox(img: Image.Image, row, margin: float = 0.1) -> Image.Image:
    x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
    dx, dy = w * margin, h * margin
    box = (max(0, x - dx), max(0, y - dy), min(img.width, x + w + dx), min(img.height, y + h + dy))
    return img.crop(tuple(int(round(v)) for v in box))


def extract_deep_features(meta: pd.DataFrame, spec: dict[str, Any], batch_size: int = 64,
                          num_workers: int = 2, device: str | None = None, pretrained: bool = True) -> np.ndarray:
    """Return an ``(N, D)`` float32 matrix of pooled embeddings for every row of ``meta``."""
    import torch
    import timm
    from timm.data import resolve_data_config
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(spec["backbone"], pretrained=pretrained, num_classes=0, **spec.get("model_kwargs", {}))
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    resize, crop = spec.get("resize", 256), spec.get("crop", 224)
    interp = T.InterpolationMode.BICUBIC if cfg.get("interpolation") == "bicubic" else T.InterpolationMode.BILINEAR
    resize_op = T.Resize((resize, resize), interpolation=interp) if spec.get("squash") else T.Resize(resize, interpolation=interp)
    transform = T.Compose([resize_op, T.CenterCrop(crop), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    use_bbox, margin = bool(spec.get("use_bbox")), spec.get("bbox_margin", 0.1)
    records = meta.reset_index(drop=True)

    class _DS(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            row = records.iloc[i]
            img = Image.open(row["path"]).convert("RGB")
            if use_bbox:
                img = _crop_bbox(img, row, margin)
            return transform(img)

    loader = DataLoader(_DS(), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    chunks = []
    use_amp = device == "cuda"
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(batch.to(device, non_blocking=True))
            chunks.append(out.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def extract_hog_features(meta: pd.DataFrame, size: int = 128, pixels_per_cell: int = 16,
                         orientations: int = 9, use_bbox: bool = False) -> np.ndarray:
    """HOG baseline on grayscale images resized to ``size x size``."""
    from skimage.feature import hog

    feats = []
    for _, row in meta.iterrows():
        img = Image.open(row["path"]).convert("L")
        if use_bbox:
            img = _crop_bbox(img, row)
        arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
        feats.append(hog(arr, orientations=orientations, pixels_per_cell=(pixels_per_cell,) * 2,
                         cells_per_block=(2, 2), feature_vector=True))
    return np.asarray(feats, dtype=np.float32)


def get_features(meta: pd.DataFrame, spec: dict[str, Any], out_dir: str | Path = "features/image",
                 batch_size: int = 64, num_workers: int = 2, force: bool = False, pretrained: bool = True) -> np.ndarray:
    """Load the cached ``.npy`` for ``spec`` or compute and save it."""
    path = Path(out_dir) / feature_filename(spec)
    if path.exists() and not force:
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec["backbone"] == "hog":
        arr = extract_hog_features(meta, spec.get("size", 128), use_bbox=spec.get("use_bbox", False))
    else:
        arr = extract_deep_features(meta, spec, batch_size=batch_size, num_workers=num_workers, pretrained=pretrained)
    np.save(path, arr)
    return arr
