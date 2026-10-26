import numpy as np
import pandas as pd
import pytest

from modules.config import deep_update, load_config
from modules.tabular.data import clean_profiles, make_target
from modules.tabular.experiments import make_pipeline
from modules.tabular.preprocessing import DropHighMissing, FrequencyEncoder, MultiHotEncoder, build_preprocessor

CFG = load_config("configs/tabular.yaml")


def fake_profiles(n=400, seed=0):
    rng = np.random.RandomState(seed)
    pick = lambda opts, p_nan=0.1: [np.nan if rng.rand() < p_nan else rng.choice(opts) for _ in range(n)]
    df = pd.DataFrame({
        "age": rng.randint(18, 60, n), "height": rng.normal(68, 4, n), "income": rng.choice([-1, 20000, 50000], n, p=[.8, .1, .1]),
        "sex": pick(["m", "f"], 0), "orientation": pick(["straight", "gay"], 0), "status": pick(["single", "seeing someone"], 0),
        "body_type": pick(["fit", "average", "thin"]), "diet": pick(["anything", "vegetarian"], 0.4),
        "offspring": pick(["has kids", "doesn&rsquo;t have kids"], 0.6), "pets": pick(["likes dogs", "likes cats"]),
        "drinks": pick(["socially", "rarely", "often"]), "smokes": pick(["no", "sometimes"]), "drugs": pick(["never", "sometimes"]),
        "education": pick(["graduated from college/university", "working on masters program", "dropped out of high school",
                           "graduated from ph.d program", "space camp"]),
        "religion": pick(["agnosticism but not too serious about it", "catholicism and laughing about it"]),
        "sign": pick(["gemini and it&rsquo;s fun to think about", "leo but it doesn&rsquo;t matter"]),
        "speaks": pick(["english (fluently), spanish (poorly)", "english"]), "ethnicity": pick(["asian, white", "white", "black"]),
        "job": pick(["computer / hardware / software", "science / tech / engineering", "medicine / health", "student", "other"], 0.1),
    })
    return df


@pytest.fixture(scope="module")
def dataset():
    d = CFG["data"]
    df = clean_profiles(fake_profiles())
    df = make_target(df, d["stem_jobs"], d["exclude_jobs"])
    y = df.pop("is_stem")
    X = df.drop(columns=["job", *d["drop_columns"]])
    return X, y


def test_cleaning_handles_disguised_missing_and_entities():
    df = clean_profiles(fake_profiles())
    assert df["income"].isna().mean() > 0.7  # -1 became NaN
    assert not df["offspring"].dropna().str.contains("&rsquo;").any()
    assert set(df["education_level"].dropna()) <= {"high school", "two-year college", "college/university", "masters", "advanced degree"}
    assert "gemini" in set(df["sign_base"].dropna())
    assert not df["speaks_langs"].dropna().str.contains(r"\(").any()


def test_target_excludes_uninformative_jobs():
    df = clean_profiles(fake_profiles())
    out = make_target(df, CFG["data"]["stem_jobs"], CFG["data"]["exclude_jobs"])
    assert not out["job"].isin(["student", "other"]).any()
    assert set(out["is_stem"].unique()) == {0, 1}


@pytest.mark.parametrize("name", list(CFG["experiments"]))
def test_every_experiment_builds_and_outputs_finite_numbers(dataset, name):
    X, y = dataset
    pre_cfg = deep_update(CFG["preprocess"], CFG["experiments"][name])
    pre = build_preprocessor(pre_cfg)
    split = len(X) // 2
    Z_train = pre.fit_transform(X.iloc[:split], y.iloc[:split])
    Z_test = pre.transform(X.iloc[split:])
    assert Z_train.shape[1] == Z_test.shape[1]
    assert np.isfinite(Z_train).all() and np.isfinite(Z_test).all()


def test_minmax_range_is_respected(dataset):
    X, y = dataset
    cfg = deep_update(CFG["preprocess"], {"scaler": "minmax", "minmax_range": [-1, 1], "nominal_encoding": "frequency"})
    Z = build_preprocessor(cfg).fit_transform(X, y)
    assert Z.min() >= -1 - 1e-9 and Z.max() <= 1 + 1e-9


def test_drop_high_missing_uses_training_fold_only():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [np.nan, np.nan, np.nan, 1]})
    t = DropHighMissing(0.5).fit(train)
    assert t.dropped_ == ["b"]
    assert list(t.transform(pd.DataFrame({"a": [1], "b": [5]})).columns) == ["a"]


def test_encoders_handle_unseen_values():
    fe = FrequencyEncoder().fit(pd.DataFrame({"c": ["x", "x", "y"]}))
    assert fe.transform(pd.DataFrame({"c": ["z", "x"]}))[:, 0].tolist() == [0.0, pytest.approx(2 / 3)]
    mh = MultiHotEncoder(min_count=2).fit(pd.Series(["a, b", "a", "b, c", np.nan]))
    assert mh.vocab_ == ["a", "b"]
    assert mh.transform(pd.Series(["a, z", np.nan])).tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_end_to_end_pipeline_fits(dataset):
    X, y = dataset
    pipe = make_pipeline(CFG["preprocess"], "logreg", CFG["models"]["logreg"])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)
