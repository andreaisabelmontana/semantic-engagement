"""Models and evaluation.

The semantic model is a pipeline: LSA embedding -> standardize -> Ridge (or
GradientBoosting) regressor predicting the engagement index. The baseline is the
same kind of regressor fit on shallow metadata only. We score both on the same
held-out split so the comparison is apples-to-apples.

R^2 and Pearson correlation are reported on held-out data. A semantic model that
truly captures the planted topic signal should clear the metadata baseline by a
wide margin.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from .features import LSAEmbedder, metadata_features


@dataclass
class SemanticModel:
    """A fitted semantic regressor (embedder + regressor) you can call."""
    embedder: LSAEmbedder
    regressor: object

    def predict(self, captions: list[str]) -> np.ndarray:
        emb = self.embedder.transform(captions)
        return self.regressor.predict(emb)


@dataclass
class Result:
    name: str
    r2: float
    pearson: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (f"{self.name:<16} R2={self.r2:+.3f}  r={self.pearson:+.3f}  "
                f"(train={self.n_train}, test={self.n_test})")


def _make_regressor(kind: str, seed: int):
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if kind == "gboost":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=seed,
        )
    raise ValueError(f"unknown regressor kind: {kind!r}")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _split_indices(n: int, test_size: float, seed: int):
    idx = np.arange(n)
    return train_test_split(idx, test_size=test_size, random_state=seed)


def train_semantic(data: dict, *, n_components: int = 64, regressor: str = "ridge",
                   seed: int = 7) -> SemanticModel:
    """Fit the semantic model on the FULL dataset (for inference/demo use)."""
    captions = data["captions"]
    y = np.asarray(data["engagement"], dtype=np.float64)

    embedder = LSAEmbedder(n_components=n_components, random_state=seed)
    emb = embedder.fit_transform(captions)

    reg = make_pipeline(StandardScaler(), _make_regressor(regressor, seed))
    reg.fit(emb, y)
    return SemanticModel(embedder=embedder, regressor=reg)


def train_metadata(data: dict, *, regressor: str = "ridge", seed: int = 7):
    """Fit the metadata-only baseline on the FULL dataset."""
    X = metadata_features(data)
    y = np.asarray(data["engagement"], dtype=np.float64)
    reg = make_pipeline(StandardScaler(), _make_regressor(regressor, seed))
    reg.fit(X, y)
    return reg


def evaluate(data: dict, *, n_components: int = 64, regressor: str = "ridge",
             test_size: float = 0.25, seed: int = 7) -> dict:
    """Held-out comparison of the semantic model vs the metadata baseline.

    Both models use the SAME train/test split. The embedder is fit on the
    training captions only (no test leakage) and applied to the test captions.

    Returns dict with keys "semantic" and "metadata", each a Result.
    """
    captions = data["captions"]
    y = np.asarray(data["engagement"], dtype=np.float64)
    n = len(captions)

    train_idx, test_idx = _split_indices(n, test_size, seed)

    cap_train = [captions[i] for i in train_idx]
    cap_test = [captions[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- semantic: fit embedder on TRAIN ONLY, then regressor ---
    embedder = LSAEmbedder(n_components=n_components, random_state=seed)
    emb_train = embedder.fit_transform(cap_train)
    emb_test = embedder.transform(cap_test)

    sem_reg = make_pipeline(StandardScaler(), _make_regressor(regressor, seed))
    sem_reg.fit(emb_train, y_train)
    sem_pred = sem_reg.predict(emb_test)
    sem_result = Result(
        name="semantic (LSA)",
        r2=float(r2_score(y_test, sem_pred)),
        pearson=_pearson(y_test, sem_pred),
        n_train=len(train_idx), n_test=len(test_idx),
    )

    # --- metadata baseline on the SAME split ---
    X = metadata_features(data)
    X_train, X_test = X[train_idx], X[test_idx]
    meta_reg = make_pipeline(StandardScaler(), _make_regressor(regressor, seed))
    meta_reg.fit(X_train, y_train)
    meta_pred = meta_reg.predict(X_test)
    meta_result = Result(
        name="metadata-only",
        r2=float(r2_score(y_test, meta_pred)),
        pearson=_pearson(y_test, meta_pred),
        n_train=len(train_idx), n_test=len(test_idx),
    )

    return {"semantic": sem_result, "metadata": meta_result}
