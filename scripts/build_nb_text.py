"""Generate notebooks/02_text.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_common import bootstrap, code, md, save

cells = [
    md("""
# Bài 2 - Text: phân loại chủ đề tin tức tiếng Nhật (livedoor News Corpus)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/02_text.ipynb)

**Học máy (CO3117) - HK1 2026-2027 - HCMUT.** Cấu hình: `configs/text.yaml`.

| Mục | Nội dung |
|---|---|
| Dữ liệu | livedoor News Corpus (RONDHUIT), 7.367 bài, 9 chuyên mục; CC BY-ND 2.1 JP |
| Bài toán | Phân loại chủ đề đa lớp (9 lớp) từ tiêu đề + nội dung |
| Đặc trưng | **Truyền thống:** BoW, TF-IDF từ (MeCab), TF-IDF n-gram ký tự. **Embedding:** `multilingual-e5-base` (lưu `.npy`) |
| Mô hình | Logistic Regression, Linear SVM, Naive Bayes (Complement), Random Forest; RBF-SVM cho embedding |
| Thước đo | macro-F1 (StratifiedKFold, mean ± std), accuracy, thời gian |

> Khuyến nghị bật GPU (Runtime > Change runtime type > T4) để tính embedding nhanh hơn; CPU vẫn chạy được nhưng chậm.
"""),
    bootstrap(),
    code("""
%pip install -q fugashi unidic-lite sentence-transformers
"""),
    code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

from modules.config import load_config
from modules.utils import set_seed, timer
from modules.text.data import prepare_dataset, build_text, duplicate_report
from modules.text.features import JapaneseTokenizer, tokenize_corpus, embed_texts, cached_embeddings
from modules.text.experiments import iter_sparse_runs, iter_dense_runs, run_text_experiments
from modules.models import build_classifier
from modules.text.features import build_sparse_pipeline
from modules.evaluation import holdout_evaluate, save_table, format_mean_std
from modules import viz

cfg = load_config("configs/text.yaml")
set_seed(cfg["seed"])
QUICK = False   # True: 3-fold CV, fewer feature sets (smoke test)
if QUICK:
    cfg["cv_folds"] = 3
"""),
    md("## 1. Dữ liệu và EDA"),
    code("""
df, train_df, test_df = prepare_dataset(cfg)
fields = cfg["data"]["text_fields"]
print(df.shape, "| train/test:", len(train_df), len(test_df))
display(df.sample(3, random_state=0)[["category", "date", "title"]])
print(duplicate_report(df, fields))
"""),
    code("""
viz.plot_class_distribution(df["category"], "Số bài theo chuyên mục", "reports/figures/text_classes.png"); plt.show()
df["n_chars"] = build_text(df, fields).str.len()
df.boxplot(column="n_chars", by="category", rot=45, figsize=(8, 4)); plt.suptitle(""); plt.title("Độ dài văn bản (ký tự)")
plt.tight_layout(); plt.savefig("reports/figures/text_length.png", dpi=150); plt.show()
print("Imbalance ratio (max/min):", round(df["category"].value_counts().max() / df["category"].value_counts().min(), 2))
"""),
    md("""
**Nhận xét EDA (điền sau):** các lớp lệch nhẹ (~1,76 lần) -> dùng macro-F1; độ dài bài khác nhau theo chuyên mục; kiểm tra trùng lặp giữa train/test để loại rò rỉ.
"""),
    md("""
## 2. Tiền xử lý văn bản
Tiếng Nhật không có khoảng trắng giữa từ nên cần tách từ bằng MeCab (`fugashi` + UniDic-lite). Việc tách từ **không có tham số học** nên có thể làm một lần trước khi chia fold mà không rò rỉ. Ta giữ danh từ, động từ, tính từ (cấu hình `tokenizer.pos`); đặc trưng n-gram **ký tự** không cần tokenizer và dùng làm mốc so sánh.
"""),
    code("""
tok = JapaneseTokenizer(backend=cfg["tokenizer"]["backend"], pos=cfg["tokenizer"]["pos"], lemma=cfg["tokenizer"]["lemma"])
sample = build_text(train_df.head(1), fields).iloc[0][:120]
print(sample); print(tokenize_corpus([sample], tok)[0])
"""),
    code("""
train_raw, test_raw = build_text(train_df, fields).tolist(), build_text(test_df, fields).tolist()
with timer() as t: train_tok = tokenize_corpus(train_raw, tok)
test_tok = tokenize_corpus(test_raw, tok)
print(f"Tokenised {len(train_raw) + len(test_raw)} docs; train took {t.seconds:.1f}s")
y_train, y_test = train_df["category"].to_numpy(), test_df["category"].to_numpy()
labels = sorted(df["category"].unique())
"""),
    md("""
## 3. Phương pháp 1 - đặc trưng truyền thống (BoW / TF-IDF)
Vectorizer nằm **trong** Pipeline cùng bộ phân loại nên vốn từ vựng và IDF chỉ học từ fold huấn luyện.
"""),
    code("""
feat_subset = ["tfidf_word_1_2", "tfidf_char_1_3"] if QUICK else None
model_subset = ["logreg", "svm_linear"] if QUICK else None
sparse = run_text_experiments(cfg, iter_sparse_runs(cfg, train_raw, train_tok, y_train, feat_subset, model_subset))
display(sparse.round(4))
"""),
    md("""
## 4. Phương pháp 2 - embedding tiền huấn luyện
`multilingual-e5-base` hỗ trợ tiếng Nhật, không cần tách từ. Embedding được tính một lần cho tập train và test, lưu `.npy` trong `features/text/` và tái sử dụng khi chạy lại.
"""),
    code("""
ecfg = cfg["embeddings"]["e5_base"]
slug = ecfg["model"].replace("/", "-")
emb = {}
for name, texts in (("train", train_raw), ("test", test_raw)):
    emb[name] = cached_embeddings(Path(f"features/text/{slug}__{name}.npy"),
        lambda texts=texts: embed_texts(texts, ecfg["model"], ecfg["prefix"], ecfg["batch_size"], ecfg["max_seq_length"]))
    print(name, emb[name].shape)
np.save("features/text/y_train.npy", y_train); np.save("features/text/y_test.npy", y_test)
"""),
    code("""
dense_models = ["logreg", "svm_linear"] if QUICK else None
dense = run_text_experiments(cfg, iter_dense_runs(cfg, {"e5_base": emb["train"]}, y_train, dense_models))
display(dense.round(4))
"""),
    md("## 5. So sánh và phân tích"),
    code("""
results = pd.concat([sparse, dense], ignore_index=True)
save_table(results, "reports/results/text_cv_results.csv")
ok = results[results["status"] == "ok"].copy()
ok["macro_f1"] = format_mean_std(ok, "macro_f1"); ok["fit_s"] = ok["fit_time_s"].round(2); ok["predict_s"] = ok["predict_time_s"].round(3)
display(ok.pivot(index="experiment", columns="model", values="macro_f1"))
viz.plot_results(results, title="macro-F1 (CV, mean ± std)", save_path="reports/figures/text_cv_comparison.png"); plt.show()
display(ok[["experiment", "model", "macro_f1", "fit_s", "predict_s"]].sort_values("macro_f1", ascending=False).head(10))
"""),
    md("""
**Phân tích (điền sau):** (1) TF-IDF từ so với n-gram ký tự (không cần tokenizer); (2) embedding có thắng TF-IDF không, đổi lại chi phí tính embedding; (3) NB nhanh nhưng kém?; (4) RF trên SVD so với tuyến tính; (5) lớp nào khó nhất.
"""),
    md("## 6. Đánh giá cuối trên tập test và phân tích lỗi"),
    code("""
best = ok.sort_values("macro_f1_mean", ascending=False).iloc[0]
print("Best by CV:", best["experiment"], "/", best["model"])
if best["experiment"].startswith("emb:"):
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    clf = build_classifier(best["model"], cfg["models_dense"][best["model"]], cfg["seed"])
    est = make_pipeline(StandardScaler(), clf) if best["model"] != "random_forest" else clf
    Xtr, Xte = emb["train"], emb["test"]
else:
    params = dict(cfg["models_sparse"][best["model"]]); svd = params.pop("svd_components", None)
    est = build_sparse_pipeline(cfg["features"][best["experiment"]], build_classifier(best["model"], params, cfg["seed"]), svd)
    word = cfg["features"][best["experiment"]]["type"].endswith("_word")
    Xtr, Xte = np.asarray(train_tok if word else train_raw, dtype=object), np.asarray(test_tok if word else test_raw, dtype=object)
final = holdout_evaluate(est, Xtr, y_train, Xte, y_test, labels=labels)
print(f"Test macro-F1={final['macro_f1']:.4f}  accuracy={final['accuracy']:.4f}  fit={final['fit_time_s']:.1f}s  predict={final['predict_time_s']:.2f}s")
viz.plot_confusion(final["confusion"], labels, title="Confusion matrix (test)", save_path="reports/figures/text_confusion.png"); plt.show()
display(final["report"].round(3))
"""),
    code("""
# Most confused class pairs and examples of misclassified titles
cm = final["confusion"].copy(); np.fill_diagonal(cm, 0)
pairs = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]), reverse=True)[:5]
for n, true, pred in pairs: print(f"{n:>3}  {true}  ->  {pred}")
wrong = test_df.assign(pred=final["y_pred"]).query("category != pred")[["category", "pred", "title"]]
display(wrong.head(10))
"""),
    md("""
**Phân tích lỗi (điền sau):** cặp chuyên mục hay nhầm và lý do (chồng lấn chủ đề, bài ngắn, ...); đề xuất cải thiện (fine-tune BERT tiếng Nhật - xem phần bonus).
"""),
    md("## 7. Lưu đặc trưng"),
    code("""
# Embeddings are already saved; also save a 300-d LSA (TruncatedSVD) projection of TF-IDF as .npy for quick reuse.
from modules.text.features import build_vectorizer
from sklearn.decomposition import TruncatedSVD
vec = build_vectorizer(cfg["features"]["tfidf_word_1_2"]).fit(train_tok)
svd = TruncatedSVD(300, random_state=cfg["seed"]).fit(vec.transform(train_tok))
np.save("features/text/tfidf_lsa300__train.npy", svd.transform(vec.transform(train_tok)).astype(np.float32))
np.save("features/text/tfidf_lsa300__test.npy", svd.transform(vec.transform(test_tok)).astype(np.float32))
print(sorted(p.name for p in Path("features/text").iterdir()))
"""),
]

save(cells, str(Path(__file__).parents[1] / "notebooks" / "02_text.ipynb"), "02_text")
