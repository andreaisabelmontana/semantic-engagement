# Semantic Engagement

Predict short-video engagement from the **meaning** of a caption — using
sentence embeddings, not just hashtag counts and length.

The question: does *what a caption says* carry engagement signal that shallow
metadata (length, number of hashtags, number of emoji) misses? This repo answers
it on a controlled synthetic dataset where the answer is knowable, and shows the
semantic model winning by a wide margin.

- **Live site:** https://andreaisabelmontana.github.io/semantic-engagement/

## Result (real, held-out)

Held-out 25% test split, 1200 synthetic captions, seed 7, Ridge regressor:

| Model | Held-out R² | Pearson r |
|---|---|---|
| **Semantic (LSA embeddings)** | **+0.688** | **+0.833** |
| Metadata-only baseline | −0.014 | −0.008 |

The semantic model explains ~69% of held-out engagement variance; the metadata
baseline explains essentially none (~0). The lift is **+0.70 R²**. The same gap
holds with a GradientBoosting regressor (semantic R² ≈ +0.68, baseline ≈ −0.07).

Reproduce:

```bash
pip install -r requirements.txt
python demo.py          # prints the numbers above + sample predictions
python -m pytest -q     # 11 tests, including "semantic beats baseline"
```

## Method

**Embeddings (the honest version).** Captions are turned into sentence
embeddings with **LSA**: a TF-IDF vectorizer (1–2 grams) followed by
**Truncated SVD** to 64 dimensions, then L2-normalized. This is fully local,
fast, deterministic, and needs no model download. It captures topical /
co-occurrence similarity — not deep compositional semantics like a transformer —
which is exactly enough to separate the themes in short captions. (The code is
structured so a transformer encoder could be dropped in as the embedder; LSA is
the default and what every number here comes from.)

**Regressor.** The embedding is standardized and fed to a **Ridge** regressor
(GradientBoosting is also supported) predicting an engagement index in 0–100.

**Baseline.** The exact same regressor, trained on metadata only — character
length, word count, hashtag count, emoji count — on the **same train/test
split**. No meaning, just surface features.

The embedder is fit on the training captions only and applied to the test
captions, so there is no test-set leakage.

## The synthetic data, and why it's a fair test

The dataset (`data/captions.csv`, committed) is **synthetic**. No real platform
data is used or implied. It's built so the ground truth is known:

- Each caption belongs to a hidden **topic** (`transformation`, `wholesome`,
  `howto`, `promo`, `rant`), each with a planted latent engagement weight — some
  kinds of meaning travel, others don't.
- Captions are composed from topic-specific vocabulary, so same-theme captions
  are semantically similar and an embedding can group them.
- **Engagement = topic weight + a small sentiment term + Gaussian noise.** It
  does **not** depend on length, hashtag count, or emoji count.
- Hashtag and emoji counts are sampled **independently of the topic**, so the
  metadata baseline has no back-door to the planted signal.

That last point is what makes the comparison fair rather than rigged: metadata
isn't sabotaged in a way that secretly encodes the topic — it's simply
uninformative, which is the hypothesis under test. A model that recovers
engagement here has to recover **meaning**, and only the semantic model can.

Everything is seeded, so the dataset and all results are reproducible
bit-for-bit.

## Package layout

```
engage/
  synth.py      synthetic caption + engagement generator (planted topic signal)
  features.py   LSAEmbedder (TF-IDF + SVD) and metadata baseline features
  model.py      train semantic / metadata regressors, held-out evaluate()
data/
  captions.csv  1200 committed synthetic captions (regenerable from synth.py)
demo.py         prints held-out R²/r for both models + sample predictions
tests/          pytest suite (semantic-beats-baseline, shapes, determinism)
```

## API sketch

```python
from engage import generate_dataset, evaluate, train_semantic

data = generate_dataset(n=1200, seed=7)
res = evaluate(data, n_components=64, regressor="ridge", seed=7)
print(res["semantic"], res["metadata"])

model = train_semantic(data, seed=7)
model.predict(["the glow up is rebuilt myself stronger you can too love"])
```

## Honest limits

- The data is synthetic by construction; real captions are messier and real
  engagement depends on far more than text (timing, audience, the video itself).
  This repo measures *whether a meaning-based model can beat a metadata baseline
  when meaning is the signal* — not real-world virality.
- LSA is a bag-of-co-occurrences method. It won't catch sarcasm, negation, or
  word order the way a transformer would. It's the honest default here, not the
  ceiling.
- Correlation isn't causation: a theme that *predicts* engagement here isn't a
  recipe to *cause* it.

## License

MIT — see [LICENSE](LICENSE).
