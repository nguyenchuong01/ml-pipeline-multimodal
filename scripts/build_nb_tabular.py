"""Generate notebooks/01_tabular.ipynb (run: python scripts/build_nb_tabular.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_common import bootstrap, code, md, save

cells = [
    md("""
# Bài 1 - Tabular: dự đoán nhóm nghề STEM từ hồ sơ OkCupid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/01_tabular.ipynb)

**Học máy (CO3117) - HK1 2026-2027 - HCMUT.** Notebook chạy theo thứ tự *Run all*; toàn bộ thiết lập nằm trong `configs/tabular.yaml`.

| Mục | Nội dung |
|---|---|
| Dữ liệu | OkCupid Profile Data (Kim & Escobedo-Land, 2015), 59.946 hồ sơ, 19 cột, tải công khai từ GitHub của tác giả |
| Bài toán | Phân loại nhị phân: nghề thuộc nhóm STEM (`science / tech / engineering`, `computer / hardware / software`) hay các nhóm nghề khác |
| Thước đo | macro-F1 (chính), accuracy; StratifiedKFold, báo cáo mean ± std; thời gian huấn luyện / dự đoán |
| Pipeline | Làm sạch -> xử lý thiếu -> mã hóa biến phân loại -> chuẩn hóa -> (PCA) -> LR / SVM / Random Forest |

> Lưu ý đạo đức: dữ liệu gồm hồ sơ công khai đã được tác giả công bố cho mục đích giảng dạy; ta chỉ phân tích gộp, không truy vết cá nhân.
"""),
    bootstrap(),
    code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import train_test_split

from modules.config import load_config, deep_update
from modules.utils import set_seed
from modules.tabular.data import load_raw, clean_profiles, prepare_dataset
from modules.tabular.experiments import make_pipeline, run_experiments
from modules.evaluation import holdout_evaluate, save_table, format_mean_std
from modules import viz

cfg = load_config("configs/tabular.yaml")
set_seed(cfg["seed"])
set_config(display="diagram")
QUICK = False   # True: 3-fold CV and fewer models, for a fast smoke test
if QUICK:
    cfg["cv_folds"] = 3
print({k: cfg[k] for k in ("seed", "cv_folds")})
"""),
    md("## 1. Dữ liệu và EDA"),
    code("""
raw = load_raw(cfg["data"]["url"], cfg["data"]["dir"])
print("Raw shape:", raw.shape)
display(raw.head(3).T)
"""),
    md("""
### 1.1 Giá trị thiếu, kể cả thiếu "ngụy trang"
Cột `income` dùng `-1` để biểu thị *không khai báo*: nếu để nguyên, `-1` sẽ bị coi là một mức thu nhập thật. Ta chuyển thành NaN trước khi tính tỉ lệ thiếu.
"""),
    code("""
print(f"income == -1 (disguised missing): {(raw['income'] == -1).mean():.1%}")
clean = clean_profiles(raw, cfg["data"]["income_missing_value"])
viz.plot_missing(clean, top=15, save_path="reports/figures/tabular_missing.png"); plt.show()
"""),
    md("""
### 1.2 Xây dựng nhãn và loại rò rỉ nhãn
Nhãn `is_stem` được suy ra từ cột `job`, nên `job` bị **loại khỏi đặc trưng** (rò rỉ nhãn). Các nhãn nghề không cung cấp thông tin (`other`, `student`, `rather not say`, `unemployed`, `retired`) và hồ sơ không khai báo nghề bị bỏ. Sau đó dữ liệu được chia 80% huấn luyện / 20% kiểm tra (stratified); từ đây EDA và thí nghiệm chỉ dùng 80% huấn luyện.
"""),
    code("""
X_all, y_all = prepare_dataset(cfg)
# Split FIRST: the hold-out test set is never used for EDA, cross-validation or model selection.
X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=cfg["holdout_size"], stratify=y_all, random_state=cfg["seed"])
print("All:", X_all.shape, "| train (EDA + CV):", X.shape, "| hold-out test:", X_test.shape)
print("positives (STEM) - train:", f"{y.mean():.1%}", "| test:", f"{y_test.mean():.1%}")
print("Columns:", list(X.columns))
viz.plot_class_distribution(y.map({0: "non-STEM", 1: "STEM"}), "Phân bố nhãn", "reports/figures/tabular_target.png", horizontal=False); plt.show()
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes[:2], ["age", "height"]):
    for label, name in [(0, "non-STEM"), (1, "STEM")]:
        X.loc[y == label, col].plot(kind="hist", bins=40, alpha=0.5, density=True, ax=ax, label=name)
    ax.set_title(col); ax.legend()
rate = pd.concat([X["education_level"], y], axis=1).groupby("education_level")["is_stem"].mean()
rate.reindex(cfg["preprocess"]["ordinal"]["education_level"]).plot(kind="bar", ax=axes[2], color="#4C78A8")
axes[2].set_title("Tỉ lệ STEM theo trình độ học vấn"); plt.tight_layout()
plt.savefig("reports/figures/tabular_eda.png", dpi=150); plt.show()
print(pd.concat([X["sex"], y], axis=1).groupby("sex")["is_stem"].mean().round(3))
"""),
    md("""
**Nhận xét EDA (điền sau khi xem biểu đồ):** phân bố nhãn lệch (~25% STEM) nên dùng macro-F1 và `class_weight="balanced"`; các cột `offspring`, `income`, `diet` thiếu nhiều; ...
"""),
    md("""
## 2. Pipeline tiền xử lý
Mọi bước (impute, mã hóa, chuẩn hóa, PCA) nằm trong **một** `sklearn.Pipeline` cùng bộ phân loại, nên khi cross-validation chỉ được *fit trên fold huấn luyện* -> không rò rỉ thống kê từ fold kiểm tra.

| Nhóm cột | Xử lý (baseline) |
|---|---|
| Số (`age`, `height`, `income`) | median impute + cờ thiếu -> StandardScaler |
| Thứ bậc (`education_level`, `drinks`, `smokes`, `drugs`) | map sang thứ hạng -> median impute + cờ thiếu -> scale |
| Danh nghĩa (`sex`, `body_type`, `diet`, ...) | giá trị `missing` riêng -> One-Hot (gộp loại hiếm) |
| Đa nhãn (`ethnicity`, `speaks_langs`) | Multi-hot, bỏ token hiếm |
"""),
    code("""
display(make_pipeline(cfg["preprocess"], "logreg", cfg["models"]["logreg"], cfg["seed"]))
"""),
    md("""
## 3. Thí nghiệm
Mỗi thí nghiệm thay **một** yếu tố so với baseline (xem `experiments:` trong YAML): chiến lược xử lý thiếu, cách mã hóa, chuẩn hóa, PCA. Mỗi cấu hình chạy với 4 mô hình.
"""),
    code("""
models = ["logreg", "svm_linear"] if QUICK else None
results = run_experiments(cfg, X, y, models=models)
save_table(results, "reports/results/tabular_cv_results.csv")
"""),
    code("""
ok = results[results["status"] == "ok"].copy()
ok["macro_f1"] = format_mean_std(ok, "macro_f1")
ok["accuracy"] = format_mean_std(ok, "accuracy")
ok["fit_s"] = ok["fit_time_s"].round(2); ok["predict_s"] = ok["predict_time_s"].round(3)
table = ok.pivot(index="experiment", columns="model", values="macro_f1").reindex(list(cfg["experiments"]))
display(table)
print("Failed runs:", (results["status"] != "ok").sum())
viz.plot_results(results, title="macro-F1 (CV, mean ± std)", save_path="reports/figures/tabular_cv_comparison.png"); plt.show()
display(ok[["experiment", "model", "macro_f1", "accuracy", "fit_s", "predict_s"]].sort_values("macro_f1", ascending=False).head(10))
"""),
    md("""
**Phân tích (điền sau khi có số liệu):** (1) chiến lược xử lý thiếu nào tốt nhất và vì sao (cờ thiếu có mang thông tin?); (2) One-Hot so với frequency / ordinal-code; (3) scale có ảnh hưởng LR/SVM nhưng không ảnh hưởng RF?; (4) PCA giữ 90/95% phương sai làm tăng hay giảm macro-F1, đổi lại thời gian; (5) mô hình nào cân bằng tốt giữa chất lượng và thời gian.
"""),
    md("""
## 4. Đánh giá cuối trên tập kiểm tra giữ lại (20%)
Tập này được tách **ngay sau khi dựng nhãn** (mục 1.2) và không tham gia EDA, cross-validation hay việc chọn cấu hình. Cấu hình tốt nhất theo CV (chỉ trên 80% huấn luyện) được huấn luyện lại trên toàn bộ phần huấn luyện rồi đánh giá đúng một lần trên tập này, nên điểm hold-out là ước lượng không lạc quan.
"""),
    code("""
best = ok.sort_values("macro_f1_mean", ascending=False).iloc[0]
print("Best by CV:", best["experiment"], "/", best["model"])
pre_cfg = deep_update(cfg["preprocess"], cfg["experiments"][best["experiment"]])
X_tr, X_te, y_tr, y_te = X, X_test, y, y_test   # train part (used for CV) and untouched hold-out
final = holdout_evaluate(make_pipeline(pre_cfg, best["model"], cfg["models"][best["model"]], cfg["seed"]), X_tr, y_tr, X_te, y_te, labels=[0, 1])
print(f"Hold-out macro-F1={final['macro_f1']:.4f}  accuracy={final['accuracy']:.4f}  fit={final['fit_time_s']:.1f}s  predict={final['predict_time_s']:.2f}s")
viz.plot_confusion(final["confusion"], ["non-STEM", "STEM"], title="Confusion matrix (hold-out)", save_path="reports/figures/tabular_confusion.png"); plt.show()
display(final["report"].round(3))
"""),
    code("""
# Which features matter? Absolute logistic-regression coefficients of a dedicated LR on the best preprocessing.
lr = make_pipeline(pre_cfg, "logreg", cfg["models"]["logreg"], cfg["seed"]).fit(X_tr, y_tr)
names = lr.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
coef = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values(key=np.abs, ascending=False).head(15)
coef.iloc[::-1].plot(kind="barh", figsize=(9, 5), color=np.where(coef.iloc[::-1] > 0, "#59A14F", "#E15759"))
plt.title("Top |coef| (green = towards STEM)"); plt.tight_layout()
plt.savefig("reports/figures/tabular_coefficients.png", dpi=150); plt.show()
"""),
    md("""
**Phân tích lỗi (điền sau):** nhóm hồ sơ nào bị nhầm nhiều nhất; đặc trưng nào có hệ số lớn và có hợp lý không (ví dụ `sex`, `education_level`). Lưu ý: cột `speaks` có người khai cả ngôn ngữ lập trình (`c++`, `lisp`), đây là tín hiệu STEM rất mạnh nhưng là *tự khai báo* - cần thảo luận trong báo cáo xem có nên xem là rò rỉ hay không (có thể chạy thêm một thí nghiệm bỏ `speaks_langs`). Giới hạn của dữ liệu: mẫu tình nguyện, khu vực San Francisco, nhãn tự khai báo.
"""),
    md("## 5. Lưu đặc trưng (.npy) và kết quả"),
    code("""
from pathlib import Path
import json
out = Path("features/tabular"); out.mkdir(parents=True, exist_ok=True)
pipe = make_pipeline(pre_cfg, best["model"], cfg["models"][best["model"]], cfg["seed"])
pre = pipe.named_steps["preprocess"].fit(X_tr, y_tr)
np.save(out / "X_train.npy", pre.transform(X_tr).astype(np.float32)); np.save(out / "y_train.npy", y_tr.to_numpy())
np.save(out / "X_test.npy", pre.transform(X_te).astype(np.float32));  np.save(out / "y_test.npy", y_te.to_numpy())
last = pre.steps[-1][1]
names = pre.named_steps["columns"].get_feature_names_out() if last is pre.named_steps["columns"] else [f"pc{i}" for i in range(last.n_components_)]
json.dump({"experiment": best["experiment"], "feature_names": [str(n) for n in names]}, open(out / "feature_names.json", "w"), indent=1)
print({p.name: p.stat().st_size // 1024 for p in out.iterdir()}, "KB")
"""),
]

save(cells, str(Path(__file__).parents[1] / "notebooks" / "01_tabular.ipynb"), "01_tabular")
