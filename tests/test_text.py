import numpy as np
import pytest

from modules.config import load_config
from modules.evaluation import cross_validate_model
from modules.models import build_classifier
from modules.text.data import build_text, load_corpus, split_dataset
from modules.text.features import JapaneseTokenizer, build_sparse_pipeline, cached_embeddings, tokenize_corpus

CFG = load_config("configs/text.yaml")


def write_fake_corpus(root):
    topics = {"sports-watch": "野球 サッカー 試合 勝利 選手", "it-life-hack": "スマホ アプリ パソコン 設定 便利"}
    for cat, words in topics.items():
        d = root / cat
        d.mkdir(parents=True)
        for i in range(12):
            (d / f"{cat}-{i}.txt").write_text(f"http://x/{i}\n2012-01-01\n{cat}{i}の記事\n{words} {words}\n", encoding="utf-8")
        (d / "LICENSE.txt").write_text("license", encoding="utf-8")  # must be ignored
    return root


def test_corpus_parsing_and_split(tmp_path):
    df = load_corpus(write_fake_corpus(tmp_path))
    assert len(df) == 24 and set(df["category"]) == {"sports-watch", "it-life-hack"}
    assert df.loc[0, "title"].endswith("の記事") and "野球" in df.loc[0, "body"] or "スマホ" in df.loc[0, "body"]
    train, test = split_dataset(df, test_size=0.25, seed=1)
    assert len(train) + len(test) == 24 and train["category"].value_counts().nunique() == 1
    assert build_text(df, ["title", "body"]).str.contains("記事").all()


@pytest.mark.parametrize("feat", ["tfidf_char_1_3"])
@pytest.mark.parametrize("model", ["logreg", "svm_linear", "complement_nb"])
def test_char_pipeline_learns_separable_topics(tmp_path, feat, model):
    df = load_corpus(write_fake_corpus(tmp_path))
    X = np.asarray(build_text(df, ["title", "body"]), dtype=object)
    fcfg = {**CFG["features"][feat], "min_df": 1}
    pipe = build_sparse_pipeline(fcfg, build_classifier(model, CFG["models_sparse"][model]))
    res = cross_validate_model(pipe, X, df["category"].to_numpy(), n_splits=3)
    assert res["macro_f1_mean"] > 0.9


def test_fugashi_tokenizer_if_available():
    pytest.importorskip("fugashi")
    tok = JapaneseTokenizer(pos=["名詞"])
    out = tokenize_corpus(["機械学習の課題を提出します"], tok)[0].split()
    assert "課題" in out and "の" not in out


def test_char_tokenizer_fallback():
    assert JapaneseTokenizer(backend="char")("機械 学習") == list("機械学習")


def test_embedding_cache_roundtrip(tmp_path):
    calls = []
    fn = lambda: (calls.append(1), np.ones((3, 4), dtype=np.float32))[1]
    a = cached_embeddings(tmp_path / "e.npy", fn)
    b = cached_embeddings(tmp_path / "e.npy", fn)
    assert calls == [1] and np.array_equal(a, b)
