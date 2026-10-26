"""Text feature extraction: traditional (BoW / TF-IDF) and pretrained embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- tokenization


class JapaneseTokenizer:
    """MeCab (fugashi + UniDic-lite) tokenizer.

    ``pos`` keeps only the listed coarse parts of speech (e.g. noun, verb, adjective);
    ``lemma`` returns dictionary forms. The tagger is created lazily so the object
    stays picklable. ``backend='char'`` falls back to single characters.
    """

    def __init__(self, backend: str = "fugashi", pos: Sequence[str] | None = None, lemma: bool = False):
        self.backend = backend
        self.pos = list(pos) if pos else None
        self.lemma = lemma
        self._tagger = None

    def _get_tagger(self):
        if self._tagger is None:
            import fugashi

            self._tagger = fugashi.Tagger()
        return self._tagger

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def __call__(self, text: str) -> list[str]:
        if self.backend == "char":
            return [c for c in text if not c.isspace()]
        tokens = []
        for w in self._get_tagger()(text):
            if self.pos and w.feature.pos1 not in self.pos:
                continue
            surface = w.feature.lemma if self.lemma and getattr(w.feature, "lemma", None) else w.surface
            tokens.append(surface)
        return tokens


def tokenize_corpus(texts: Sequence[str], tokenizer: JapaneseTokenizer) -> list[str]:
    """Tokenise once and join with spaces so vectorizers do not re-run MeCab in every CV fold.

    Tokenisation is stateless (no fitted parameters), therefore doing it before the
    train/test split does not leak information.
    """
    return [" ".join(tokenizer(t)) for t in texts]


# --------------------------------------------------------------------------- traditional features


def build_vectorizer(cfg: dict[str, Any]):
    """Create a vectorizer from a ``features.<name>`` config block.

    ``type``: ``tfidf_word`` | ``bow_word`` (expect pre-tokenised, space-joined text) |
    ``tfidf_char`` | ``bow_char`` (raw text, character n-grams, no tokenizer needed).
    """
    kind = cfg["type"]
    ngram = tuple(cfg.get("ngram_range", (1, 1)))
    common = dict(min_df=cfg.get("min_df", 2), max_df=cfg.get("max_df", 1.0), max_features=cfg.get("max_features"),
                  lowercase=cfg.get("lowercase", True), dtype=np.float32)
    if kind in ("tfidf_word", "bow_word"):
        kwargs = dict(analyzer="word", token_pattern=r"\S+", ngram_range=ngram, **common)
    elif kind in ("tfidf_char", "bow_char"):
        kwargs = dict(analyzer="char", ngram_range=ngram, **common)
    else:
        raise ValueError(f"Unknown vectorizer type '{kind}'")
    if kind.startswith("tfidf"):
        return TfidfVectorizer(sublinear_tf=cfg.get("sublinear_tf", True), norm=cfg.get("norm", "l2"), **kwargs)
    return CountVectorizer(**kwargs)


def build_sparse_pipeline(feature_cfg: dict[str, Any], classifier, svd_components: int | None = None) -> Pipeline:
    """Vectorizer (+ optional TruncatedSVD for dense-only learners) + classifier in one pipeline."""
    steps: list[tuple[str, Any]] = [("vec", build_vectorizer(feature_cfg))]
    if svd_components:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=0)))
    steps.append(("model", classifier))
    return Pipeline(steps)


# --------------------------------------------------------------------------- embeddings


def embed_texts(texts: Sequence[str], model_name: str, prefix: str = "", batch_size: int = 32,
                max_seq_length: int = 256, normalize: bool = True, device: str | None = None) -> np.ndarray:
    """Encode texts with a pretrained sentence-transformers model (no fine-tuning).

    For the E5 family, pass ``prefix='passage: '`` as recommended by the model card.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    inputs = [prefix + t for t in texts]
    emb = model.encode(inputs, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=normalize, convert_to_numpy=True)
    return emb.astype(np.float32)


def cached_embeddings(path: str | Path, compute_fn, force: bool = False) -> np.ndarray:
    """Load embeddings from ``path`` (.npy) or compute and save them."""
    path = Path(path)
    if path.exists() and not force:
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = compute_fn()
    np.save(path, arr)
    return arr
