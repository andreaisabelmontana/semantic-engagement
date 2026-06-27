"""Tests for the semantic-engagement pipeline.

The headline test is the claim of the whole project: a semantic model trained on
LSA caption embeddings beats a metadata-only baseline on held-out engagement,
because engagement was planted to depend on meaning. We also test embedding
shape/dimensionality and determinism under a fixed seed.
"""

import numpy as np
import pytest

from engage import (
    generate_dataset,
    evaluate,
    LSAEmbedder,
    metadata_features,
    METADATA_COLUMNS,
    train_semantic,
)


# ---------------------------------------------------------------- fixtures ---

@pytest.fixture(scope="module")
def data():
    return generate_dataset(n=1200, seed=7)


# ----------------------------------------------------------- core claim ------

def test_semantic_beats_metadata_baseline(data):
    """Semantic LSA model must clear the metadata-only baseline on held-out R^2."""
    res = evaluate(data, n_components=64, regressor="ridge", test_size=0.25, seed=7)
    sem = res["semantic"].r2
    meta = res["metadata"].r2

    # Semantic should explain a large share of variance...
    assert sem > 0.5, f"semantic R2 too low: {sem:.3f}"
    # ...and the metadata baseline should explain almost none (no planted signal).
    assert meta < 0.2, f"metadata baseline unexpectedly high: {meta:.3f}"
    # ...and the gap should be clearly in the semantic model's favour.
    assert sem - meta > 0.3, f"semantic lift too small: {sem - meta:.3f}"


def test_semantic_correlation_is_strong(data):
    res = evaluate(data, n_components=64, regressor="ridge", seed=7)
    assert res["semantic"].pearson > 0.7
    # Baseline correlation should be near zero (planted to be uninformative).
    assert abs(res["metadata"].pearson) < 0.25


def test_gboost_regressor_also_beats_baseline(data):
    """The result is not specific to Ridge; GradientBoosting wins too."""
    res = evaluate(data, n_components=64, regressor="gboost", seed=7)
    assert res["semantic"].r2 - res["metadata"].r2 > 0.3


# ----------------------------------------------- embedding shape / dims ------

def test_embedding_shape_and_dimensionality(data):
    caps = data["captions"]
    n_comp = 48
    emb = LSAEmbedder(n_components=n_comp, random_state=7).fit_transform(caps)

    assert emb.shape == (len(caps), n_comp)
    assert emb.dtype == np.float64
    assert np.isfinite(emb).all()


def test_embeddings_are_l2_normalized(data):
    caps = data["captions"]
    emb = LSAEmbedder(n_components=32, random_state=7).fit_transform(caps)
    norms = np.linalg.norm(emb, axis=1)
    # Normalizer makes each row unit length (zero-vector rows are possible but
    # not expected on this dataset).
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_metadata_feature_shape(data):
    X = metadata_features(data)
    assert X.shape == (len(data["captions"]), len(METADATA_COLUMNS))
    assert (X >= 0).all()


# --------------------------------------------------------- determinism -------

def test_dataset_generation_is_deterministic():
    a = generate_dataset(n=300, seed=42)
    b = generate_dataset(n=300, seed=42)
    assert a["captions"] == b["captions"]
    assert a["engagement"] == b["engagement"]
    assert a["topic"] == b["topic"]


def test_different_seeds_differ():
    a = generate_dataset(n=300, seed=1)
    b = generate_dataset(n=300, seed=2)
    assert a["captions"] != b["captions"]


def test_embedding_is_deterministic(data):
    caps = data["captions"]
    e1 = LSAEmbedder(n_components=40, random_state=7).fit_transform(caps)
    e2 = LSAEmbedder(n_components=40, random_state=7).fit_transform(caps)
    assert np.allclose(e1, e2)


def test_evaluate_is_deterministic(data):
    r1 = evaluate(data, n_components=64, regressor="ridge", seed=7)
    r2 = evaluate(data, n_components=64, regressor="ridge", seed=7)
    assert r1["semantic"].r2 == r2["semantic"].r2
    assert r1["metadata"].r2 == r2["metadata"].r2


# ----------------------------------------------- end-to-end inference --------

def test_predictions_respect_planted_topic_order(data):
    """Transformation/wholesome captions should outscore promo/rant captions."""
    model = train_semantic(data, n_components=64, regressor="ridge", seed=7)

    high = "the glow up is rebuilt myself stronger you can too love"
    low = "limited drop twenty percent off everything storewide shop now"
    rant = "unpopular opinion but slow walkers blocking the whole aisle end of rant"

    p_high, p_low, p_rant = model.predict([high, low, rant])
    assert p_high > p_low
    assert p_high > p_rant
