"""Generate notebooks/03_image.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_common import bootstrap, code, md, save

cells = [
    md("""
# Bài 3 - Image: phân loại 200 loài chim (CUB-200-2011) bằng đặc trưng từ backbone tiền huấn luyện

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/03_image.ipynb)

**Học máy (CO3117) - HK1 2026-2027 - HCMUT.** Cấu hình: `configs/image.yaml`. **Bật GPU** (Runtime > Change runtime type > T4).

| Mục | Nội dung |
|---|---|
| Dữ liệu | CUB-200-2011 (Wah et al., 2011): 11.788 ảnh, 200 loài; tách chuẩn 5.994 train / 5.794 test |
| Đặc trưng | Backbone đóng băng (ResNet50, EfficientNet-B0, ViT-B/16, DINOv2-S) -> embedding đã pool, lưu `.npy`; mốc truyền thống: HOG + PCA |
| Yếu tố cấu hình | backbone, `resize`, `crop`, cắt theo bounding box |
| Bộ phân loại | Logistic Regression, Linear SVM (SGD, hinge loss), Random Forest (sklearn) trên đặc trưng chuẩn hóa |
| Thước đo | macro-F1 và accuracy; CV trên train, đánh giá cuối trên tập test chuẩn |

> **Hạn chế cần nêu trong báo cáo:** ảnh CUB trùng một phần với ImageNet nên backbone pretrained trên ImageNet có thể đã thấy ảnh kiểm tra; kết quả vì vậy lạc quan hơn so với miền dữ liệu hoàn toàn mới.
"""),
    bootstrap(),
    code("""
%pip install -q timm
"""),
    code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as patches
from pathlib import Path
from PIL import Image

from modules.config import load_config
from modules.utils import set_seed, timer
from modules.image.data import prepare_dataset
from modules.image.features import feature_filename, get_features
from modules.image.experiments import compute_all_features, run_image_experiments, make_estimator
from modules.evaluation import holdout_evaluate, save_table, format_mean_std
from modules import viz

cfg = load_config("configs/image.yaml")
set_seed(cfg["seed"])
QUICK = False   # True: only 2 experiments and 2 classifiers (smoke test)
if QUICK:
    cfg["experiments"] = {k: cfg["experiments"][k] for k in ("hog_baseline", "resnet50_full")}
    cfg["models"] = {k: cfg["models"][k] for k in ("logreg", "svm_linear_sgd")}
import torch; print("CUDA:", torch.cuda.is_available())
"""),
    md("## 1. Dữ liệu và EDA"),
    code("""
meta = prepare_dataset(cfg)
train = meta["is_train"] == 1
print(meta.shape, "| train:", int(train.sum()), "| test:", int((~train).sum()), "| classes:", meta["class_id"].nunique())
display(meta.head(3))
per_class = meta[train].groupby("class_id").size()
print("Train images per class: min", per_class.min(), "max", per_class.max(), "mean", round(per_class.mean(), 1))
"""),
    code("""
sizes = pd.DataFrame([Image.open(p).size for p in meta["path"].sample(500, random_state=0)], columns=["w", "h"])
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
sizes["w"].plot(kind="hist", bins=30, ax=axes[0], title="Chiều rộng (px)"); sizes["h"].plot(kind="hist", bins=30, ax=axes[1], title="Chiều cao (px)")
per_class.plot(kind="hist", bins=15, ax=axes[2], title="Ảnh train / lớp"); plt.tight_layout()
plt.savefig("reports/figures/image_eda_sizes.png", dpi=150); plt.show()
print(sizes.describe().loc[["min", "50%", "max"]])
"""),
    code("""
# Sample images with the official bounding boxes
rng = np.random.RandomState(0); idx = rng.choice(len(meta), 8, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, i in zip(axes.ravel(), idx):
    r = meta.iloc[i]; ax.imshow(Image.open(r["path"])); ax.axis("off"); ax.set_title(r["class_name"], fontsize=9)
    ax.add_patch(patches.Rectangle((r["bbox_x"], r["bbox_y"]), r["bbox_w"], r["bbox_h"], fill=False, color="lime", lw=2))
plt.tight_layout(); plt.savefig("reports/figures/image_samples.png", dpi=150); plt.show()
"""),
    md("""
**Nhận xét EDA (điền sau):** 200 lớp cân bằng (~30 ảnh train/lớp); kích thước ảnh không đồng nhất nên bước `resize + center-crop` là tham số quan trọng; chim chiếm phần nhỏ của khung hình ở nhiều ảnh nên cắt theo bounding box có thể giúp.
"""),
    md("""
## 2. Trích đặc trưng (backbone đóng băng)
Mỗi thí nghiệm = một cấu hình trích đặc trưng trong YAML. Kết quả lưu ở `features/image/*.npy` (hàng *i* ứng với hàng *i* của bảng metadata) và được **dùng lại** nếu file đã tồn tại.
"""),
    code("""
import time
t0 = time.perf_counter()
feats = compute_all_features(cfg, meta)
print(f"Feature extraction total: {time.perf_counter() - t0:.0f}s")
for name, f in feats.items():
    print(f"{name:<22} {str(f.shape):<14} -> {feature_filename(cfg['experiments'][name]['extractor'])}")
np.save(Path(cfg["features_dir"]) / "labels.npy", meta["class_id"].to_numpy()); np.save(Path(cfg["features_dir"]) / "is_train.npy", meta["is_train"].to_numpy())
"""),
    md("""
## 3. Phân loại và so sánh (CV trên tập train chuẩn)
`StandardScaler` (+ PCA cho HOG) và bộ phân loại nằm trong cùng một pipeline nên chỉ fit trên fold huấn luyện.
"""),
    code("""
results = run_image_experiments(cfg, feats, meta)
save_table(results, "reports/results/image_cv_results.csv")
ok = results[results["status"] == "ok"].copy()
ok["macro_f1"] = format_mean_std(ok, "macro_f1"); ok["fit_s"] = ok["fit_time_s"].round(1); ok["predict_s"] = ok["predict_time_s"].round(2)
display(ok.pivot(index="experiment", columns="model", values="macro_f1").reindex(list(cfg["experiments"])))
viz.plot_results(results, title="macro-F1 (3-fold CV on train, mean ± std)", save_path="reports/figures/image_cv_comparison.png"); plt.show()
display(ok[["experiment", "model", "macro_f1", "fit_s", "predict_s"]].sort_values("macro_f1", ascending=False).head(10))
"""),
    md("""
**Phân tích (điền sau):** (1) HOG so với đặc trưng sâu: khoảng cách; (2) CNN so với ViT/DINOv2; (3) cắt bounding box có cải thiện không (và lưu ý bbox là thông tin có nhãn, không có sẵn ở thực tế); (4) `resize/crop` lớn hơn đổi lấy thời gian; (5) LR / SVM / RF: chất lượng và thời gian.
"""),
    md("## 4. Đánh giá cuối trên tập test chuẩn (5.794 ảnh)"),
    code("""
best = ok.sort_values("macro_f1_mean", ascending=False).iloc[0]
print("Best by CV:", best["experiment"], "/", best["model"])
X = feats[best["experiment"]]
pca = cfg["experiments"][best["experiment"]].get("pca")
est = make_estimator(best["model"], cfg["models"][best["model"]], pca, cfg["seed"])
tr, te = (meta["is_train"] == 1).to_numpy(), (meta["is_train"] == 0).to_numpy()
final = holdout_evaluate(est, X[tr], meta.loc[tr, "class_id"], X[te], meta.loc[te, "class_id"], labels=list(range(200)))
print(f"Test macro-F1={final['macro_f1']:.4f}  accuracy={final['accuracy']:.4f}  fit={final['fit_time_s']:.1f}s  predict={final['predict_time_s']:.2f}s")
"""),
    code("""
# 200x200 confusion matrix is unreadable -> inspect the hardest classes and the most confused pairs
names = meta.drop_duplicates("class_id").sort_values("class_id")["class_name"].tolist()
rep = final["report"].iloc[:200].copy(); rep.index = names
display(rep.sort_values("f1-score").head(10).round(3))
cm = final["confusion"].copy(); np.fill_diagonal(cm, 0)
pairs = sorted(((cm[i, j], names[i], names[j]) for i, j in zip(*np.nonzero(cm))), reverse=True)[:8]
for n, a, b in pairs: print(f"{n:>3}  {a}  ->  {b}")
rep["f1-score"].plot(kind="hist", bins=20, figsize=(6, 3), title="Phân bố F1 theo lớp"); plt.tight_layout()
plt.savefig("reports/figures/image_f1_hist.png", dpi=150); plt.show()
"""),
    code("""
# 2-D view of the learned feature space for 15 classes (t-SNE)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
sel = meta["class_id"].isin(range(0, 200, 13)).to_numpy()
Z = TSNE(2, init="pca", perplexity=30, random_state=cfg["seed"]).fit_transform(StandardScaler().fit_transform(X[sel]))
plt.figure(figsize=(7, 6)); plt.scatter(Z[:, 0], Z[:, 1], c=meta.loc[sel, "class_id"], cmap="tab20", s=8)
plt.title(f"t-SNE of {best['experiment']} features (15 classes)"); plt.tight_layout()
plt.savefig("reports/figures/image_tsne.png", dpi=150); plt.show()
"""),
    md("""
**Phân tích lỗi (điền sau):** các loài hay bị nhầm thường cùng họ (ví dụ nhóm sẻ/chích/mòng biển), ảnh nền phức tạp, tư thế khuất; ảnh hưởng của việc ImageNet trùng với CUB; hướng cải thiện (fine-tuning, augmentation, đặc trưng part-based).
"""),
]

save(cells, str(Path(__file__).parents[1] / "notebooks" / "03_image.ipynb"), "03_image")
