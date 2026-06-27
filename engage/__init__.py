"""Semantic Engagement: predict short-video engagement from caption meaning.

Public API:
    generate_dataset  -- build the synthetic, signal-bearing caption dataset
    LSAEmbedder       -- TF-IDF + truncated SVD sentence embeddings (LSA)
    metadata_features -- shallow metadata baseline features
    train_semantic    -- fit the semantic (embedding) regressor
    train_metadata    -- fit the metadata-only baseline regressor
    evaluate          -- held-out R^2 / correlation for both models
"""

from .synth import TOPICS, generate_dataset
from .features import LSAEmbedder, metadata_features, METADATA_COLUMNS
from .model import (
    train_semantic,
    train_metadata,
    evaluate,
    Result,
    SemanticModel,
)

__all__ = [
    "TOPICS",
    "generate_dataset",
    "LSAEmbedder",
    "metadata_features",
    "METADATA_COLUMNS",
    "train_semantic",
    "train_metadata",
    "evaluate",
    "Result",
    "SemanticModel",
]

__version__ = "0.1.0"
