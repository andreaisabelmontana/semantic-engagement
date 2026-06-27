"""Feature extraction.

Two competing representations of the same caption:

  1. LSAEmbedder -- a sentence embedding built from TF-IDF + truncated SVD
     (Latent Semantic Analysis). This is fully local, fast, deterministic and
     needs no model download. It maps each caption into a dense `n_components`
     vector where captions with similar *meaning* (similar word usage) land
     near each other. This is the "semantic" representation.

  2. metadata_features -- the shallow baseline: how long is the caption, how
     many hashtags, how many emoji. No meaning at all.

The honest framing: LSA is not a transformer. It captures co-occurrence /
topical similarity, not deep compositional semantics. For short topical captions
that is plenty to separate themes, which is exactly what this project tests.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


METADATA_COLUMNS = ("char_len", "word_count", "n_hashtags", "n_emoji")

# Matches most common emoji ranges (enough for the synthetic emoji we plant).
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF❤️]+"
)
_HASHTAG_RE = re.compile(r"#\w+")


class LSAEmbedder:
    """TF-IDF + Truncated SVD sentence embeddings (LSA).

    Parameters
    ----------
    n_components : int
        Embedding dimensionality (number of SVD components).
    max_features : int
        Vocabulary cap for the TF-IDF stage.
    random_state : int
        Seed for the (randomized) SVD solver -- makes embeddings deterministic.
    """

    def __init__(self, n_components: int = 64, max_features: int = 2000,
                 random_state: int = 7) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        self._pipeline = None

    def _build(self) -> None:
        tfidf = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w']+\b",
        )
        svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
            n_iter=7,
        )
        # L2-normalize so dot products behave like cosine similarity.
        self._pipeline = make_pipeline(tfidf, svd, Normalizer(copy=False))

    def fit(self, captions: list[str]) -> "LSAEmbedder":
        self._build()
        self._pipeline.fit(captions)
        return self

    def transform(self, captions: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("LSAEmbedder must be fit before transform.")
        emb = self._pipeline.transform(captions)
        return np.asarray(emb, dtype=np.float64)

    def fit_transform(self, captions: list[str]) -> np.ndarray:
        return self.fit(captions).transform(captions)


def metadata_features(data: dict) -> np.ndarray:
    """Shallow metadata baseline features for each caption.

    Columns (see METADATA_COLUMNS): character length, word count, hashtag count,
    emoji count. Word count and char length are recomputed from the caption text
    so this works on any dataset dict that has `captions`; hashtag/emoji counts
    fall back to recomputation if not pre-stored.
    """
    captions = data["captions"]
    n = len(captions)

    char_len = data.get("char_len") or [len(c) for c in captions]
    word_count = [len(c.split()) for c in captions]
    n_hashtags = data.get("n_hashtags") or [len(_HASHTAG_RE.findall(c)) for c in captions]
    n_emoji = data.get("n_emoji") or [
        sum(len(m) for m in _EMOJI_RE.findall(c)) for c in captions
    ]

    feats = np.zeros((n, len(METADATA_COLUMNS)), dtype=np.float64)
    feats[:, 0] = char_len
    feats[:, 1] = word_count
    feats[:, 2] = n_hashtags
    feats[:, 3] = n_emoji
    return feats
