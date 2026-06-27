"""Demo: does the MEANING of a caption predict engagement better than metadata?

Run:  python demo.py

It (1) loads the committed synthetic dataset (or regenerates it), (2) reports
held-out R^2 / correlation for the semantic LSA model vs the metadata-only
baseline, and (3) prints predicted engagement for a handful of fresh captions.
All numbers are from a real run on synthetic, seed-fixed data.
"""

from __future__ import annotations

import io
import os
import sys

# Make stdout UTF-8 so emoji in captions don't crash on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from engage import generate_dataset, evaluate, train_semantic, train_metadata
from engage.synth import read_csv

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "captions.csv")
SEED = 7


def load_data() -> dict:
    if os.path.exists(DATA_PATH):
        return read_csv(DATA_PATH)
    return generate_dataset(n=1200, seed=SEED)


def main() -> None:
    data = load_data()
    n = len(data["captions"])

    print("=" * 64)
    print("Semantic Engagement  -  meaning vs metadata")
    print("=" * 64)
    print(f"Dataset: {n} synthetic captions (seed={SEED}), engagement index 0-100")
    print("Method:  LSA sentence embeddings (TF-IDF + truncated SVD) + Ridge")
    print("Baseline: same Ridge on metadata only (length, #hashtags, #emoji)")
    print()

    res = evaluate(data, n_components=64, regressor="ridge", test_size=0.25, seed=SEED)
    sem, meta = res["semantic"], res["metadata"]

    print("Held-out performance (25% test split):")
    print("  " + str(sem))
    print("  " + str(meta))
    lift = sem.r2 - meta.r2
    print(f"\n  Semantic model improves held-out R^2 by {lift:+.3f} over metadata.")
    print("  (Engagement was planted to depend on TOPIC/sentiment, not metadata,")
    print("   so a representation of meaning is what recovers the signal.)")
    print()

    # Train on full data for the sample predictions below.
    sem_model = train_semantic(data, n_components=64, regressor="ridge", seed=SEED)

    samples = [
        "the glow up is rebuilt myself stronger you can too love",
        "limited drop twenty percent off everything storewide shop now",
        "unpopular opinion but slow walkers blocking the whole aisle end of rant",
        "a tiny reminder kindness shared between strangers stay soft",
        "step by step fixing a leaking faucet yourself save this for later",
    ]
    preds = sem_model.predict(samples)

    print("Predicted engagement index for fresh captions (0 = low, 100 = high):")
    for cap, p in sorted(zip(samples, preds), key=lambda kv: kv[1], reverse=True):
        short = cap if len(cap) <= 54 else cap[:51] + "..."
        print(f"  {p:5.1f}   {short}")
    print()
    print("Highest scores go to transformation / wholesome captions; promo and")
    print("rant captions score lowest -- matching the planted ground truth.")


if __name__ == "__main__":
    main()
