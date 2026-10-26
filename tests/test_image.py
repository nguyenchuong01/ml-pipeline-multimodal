import numpy as np
import pandas as pd
import pytest
from PIL import Image

from modules.image.data import load_metadata
from modules.image.features import extract_hog_features, feature_filename, get_features


def write_fake_cub(root, n_classes=3, per_class=4):
    (root / "images").mkdir(parents=True)
    rows_img, rows_lab, rows_split, rows_box, rows_cls = [], [], [], [], []
    rng = np.random.RandomState(0)
    k = 1
    for c in range(1, n_classes + 1):
        name = f"{c:03d}.Bird_{c}"
        (root / "images" / name).mkdir()
        rows_cls.append(f"{c} {name}")
        for i in range(per_class):
            rel = f"{name}/img_{i}.jpg"
            Image.fromarray(rng.randint(0, 255, (60, 80, 3), dtype=np.uint8)).save(root / "images" / rel)
            rows_img.append(f"{k} {rel}"); rows_lab.append(f"{k} {c}")
            rows_split.append(f"{k} {int(i < per_class // 2)}"); rows_box.append(f"{k} 10.0 10.0 40.0 30.0")
            k += 1
    for fname, rows in [("images.txt", rows_img), ("image_class_labels.txt", rows_lab), ("train_test_split.txt", rows_split),
                        ("bounding_boxes.txt", rows_box), ("classes.txt", rows_cls)]:
        (root / fname).write_text("\n".join(rows) + "\n")
    return root


def test_metadata_matches_official_layout(tmp_path):
    meta = load_metadata(write_fake_cub(tmp_path / "CUB_200_2011"))
    assert len(meta) == 12 and meta["image_id"].is_monotonic_increasing
    assert meta["class_id"].min() == 0 and meta["class_name"].iloc[0] == "Bird 1"
    assert meta["is_train"].sum() == 6 and meta["path"].map(lambda p: __import__("os").path.exists(p)).all()


def test_feature_filename_encodes_settings():
    assert feature_filename({"backbone": "resnet50", "resize": 256, "crop": 224}) == "resnet50__r256_c224__full.npy"
    assert feature_filename({"backbone": "resnet50", "use_bbox": True}).endswith("__bbox.npy")
    assert feature_filename({"backbone": "hog", "size": 128}) == "hog__s128.npy"


def test_hog_features_and_cache(tmp_path):
    meta = load_metadata(write_fake_cub(tmp_path / "CUB_200_2011"))
    f = get_features(meta, {"backbone": "hog", "size": 64}, out_dir=tmp_path / "feat")
    assert f.shape[0] == 12 and f.ndim == 2 and (tmp_path / "feat" / "hog__s64.npy").exists()
    assert np.array_equal(f, np.load(tmp_path / "feat" / "hog__s64.npy"))


def test_deep_extractor_shapes_and_bbox_option(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("timm"); pytest.importorskip("torchvision")
    from modules.image.features import extract_deep_features

    meta = load_metadata(write_fake_cub(tmp_path / "CUB_200_2011"))
    for use_bbox in (False, True):
        spec = {"backbone": "resnet18", "resize": 64, "crop": 64, "use_bbox": use_bbox}
        f = extract_deep_features(meta, spec, batch_size=4, num_workers=0, device="cpu", pretrained=False)
        assert f.shape == (12, 512) and np.isfinite(f).all()


def test_dinov2_config_runs_with_dynamic_image_size(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("timm"); pytest.importorskip("torchvision")
    from modules.config import load_config
    from modules.image.features import extract_deep_features

    meta = load_metadata(write_fake_cub(tmp_path / "CUB_200_2011"))
    spec = dict(load_config("configs/image.yaml")["experiments"]["dinov2_s14_full"]["extractor"])
    f = extract_deep_features(meta.head(4), spec, batch_size=2, num_workers=0, device="cpu", pretrained=False)
    assert f.shape == (4, 384) and np.isfinite(f).all()
