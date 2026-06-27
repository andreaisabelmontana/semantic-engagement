"""Synthetic, signal-bearing caption dataset.

The whole point of this project is to test a claim: that the *meaning* of a
caption carries engagement signal that shallow metadata (length, hashtag count,
emoji count) does not. To test that fairly we need data where we KNOW the ground
truth, so we plant it:

  * Each caption belongs to a hidden TOPIC (e.g. "transformation", "rant",
    "wholesome"). Every topic has a latent engagement weight -- some kinds of
    meaning travel, others don't.
  * Captions are composed from topic-specific vocabulary, so two captions about
    the same theme look semantically similar (and an embedding can tell).
  * Engagement = topic_weight + small_sentiment_term + gaussian_noise.
    It does NOT depend on caption length, hashtag count or emoji count.
  * Metadata (length, #hashtags, #emoji) is sampled roughly INDEPENDENTLY of
    the topic, so the metadata baseline has almost nothing real to learn. This
    is what makes it a fair, honest test: the semantic model should win because
    the signal genuinely lives in the meaning, not because we rigged metadata to
    be useless in a way that also leaks the topic.

Everything is seeded, so the dataset is reproducible bit-for-bit.

This data is SYNTHETIC. No real platform data is used or implied. The numbers
exist only to demonstrate that a semantic model can recover planted meaning
signal that a metadata model cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import numpy as np


@dataclass(frozen=True)
class Topic:
    name: str
    weight: float          # latent engagement contribution (the planted signal)
    openers: tuple[str, ...]
    cores: tuple[str, ...]
    closers: tuple[str, ...]


# Topic vocabularies. Words are intentionally non-overlapping between topics so
# that the bag-of-words / LSA space separates them. Weights are the planted
# ground truth: "transformation" and "wholesome" travel; "rant" and "promo"
# don't.
TOPICS: tuple[Topic, ...] = (
    Topic(
        name="transformation",
        weight=2.4,
        openers=("the glow up is", "watch this become", "from broken to", "day 90 of"),
        cores=(
            "rebuilt myself stronger", "finally healed and thriving",
            "the before and after nobody believed", "small habits compounding daily",
            "proof that change is possible", "reinventing everything from scratch",
        ),
        closers=("and it changed me", "keep going", "you can too", "no going back"),
    ),
    Topic(
        name="wholesome",
        weight=1.9,
        openers=("a tiny reminder", "soft moment of", "found so much", "just some pure"),
        cores=(
            "kindness shared between strangers", "a dog reunited with its family",
            "grandma laughing at the table", "friends showing up when it mattered",
            "gentle love on an ordinary day", "comfort and warmth and home",
        ),
        closers=("hold your people close", "be gentle today", "this is everything", "stay soft"),
    ),
    Topic(
        name="howto",
        weight=0.6,
        openers=("step by step", "here is exactly how", "the simple method for", "a quick guide to"),
        cores=(
            "fixing a leaking faucet yourself", "folding a fitted sheet properly",
            "organizing a tiny kitchen efficiently", "setting up the spreadsheet correctly",
            "cleaning grout without scrubbing forever", "wiring the shelf bracket safely",
        ),
        closers=("save this for later", "try it tonight", "works every time", "thank me later"),
    ),
    Topic(
        name="promo",
        weight=-0.8,
        openers=("limited drop", "use my code", "link in bio for", "dont miss the"),
        cores=(
            "twenty percent off everything storewide", "the new collection just launched",
            "exclusive bundle while supplies last", "subscribe for the discount today",
            "biggest sale of the entire season", "buy two and get one free",
        ),
        closers=("shop now", "selling fast", "act quickly", "dont wait"),
    ),
    Topic(
        name="rant",
        weight=-1.6,
        openers=("unpopular opinion but", "im so tired of", "can we stop", "nobody asked but"),
        cores=(
            "people who chew loudly in meetings", "slow walkers blocking the whole aisle",
            "another pointless software update again", "the constant noise from the neighbors",
            "endless emails that should be calls", "traffic on a road with no reason",
        ),
        closers=("end of rant", "anyway whatever", "im done", "thats all"),
    ),
)

# Maximum absolute engagement an emoji could add IF we let it -- we keep this at
# zero so metadata truly carries no planted signal. Length and hashtags likewise
# carry no planted contribution.
_SENTIMENT_WORDS = {
    "love": 0.5, "amazing": 0.4, "beautiful": 0.4, "best": 0.3, "happy": 0.3,
    "hate": -0.5, "worst": -0.4, "awful": -0.4, "boring": -0.3, "annoying": -0.3,
}


def _compose_caption(topic: Topic, rng: np.random.Generator) -> str:
    parts = [
        rng.choice(topic.openers),
        rng.choice(topic.cores),
        rng.choice(topic.closers),
    ]
    text = " ".join(parts)

    # Occasionally sprinkle a sentiment word. It adds a small, *real* engagement
    # term (so meaning beyond topic matters a little), and it is part of the
    # text the embedder sees -- not metadata.
    sentiment = 0.0
    if rng.random() < 0.5:
        word = rng.choice(list(_SENTIMENT_WORDS.keys()))
        sentiment = _SENTIMENT_WORDS[word]
        text = f"{text} {word}"

    return text, sentiment


def _add_surface_noise(text: str, rng: np.random.Generator) -> tuple[str, int, int]:
    """Attach hashtags and emoji to a caption.

    Crucially the COUNTS are sampled independently of the topic, so metadata
    cannot back-door the planted signal. We return the decorated text plus the
    hashtag and emoji counts so the caller can record honest metadata.
    """
    n_tags = int(rng.integers(0, 6))
    n_emoji = int(rng.integers(0, 5))

    tag_pool = ("fyp", "viral", "foryou", "trending", "explore", "reels", "shorts", "daily")
    emoji_pool = ("🔥", "✨", "😂", "❤️", "🙌", "👀", "💯", "😭")

    tags = " ".join("#" + rng.choice(tag_pool) for _ in range(n_tags))
    emojis = "".join(rng.choice(emoji_pool) for _ in range(n_emoji))

    decorated = text
    if tags:
        decorated = f"{decorated} {tags}"
    if emojis:
        decorated = f"{decorated} {emojis}"
    return decorated, n_tags, n_emoji


def generate_dataset(n: int = 1200, seed: int = 7) -> dict:
    """Generate the synthetic caption -> engagement dataset.

    Returns a dict with parallel lists:
        captions       list[str]   the decorated caption text
        engagement     list[float] the (standardized-ish) engagement target
        topic          list[str]   the hidden topic label (for inspection only)
        n_hashtags     list[int]
        n_emoji        list[int]
        char_len       list[int]

    Engagement is built from the planted topic weight + a small sentiment term +
    gaussian noise, and is then shifted/scaled to a friendly positive range. It
    is deterministic for a given (n, seed).
    """
    rng = np.random.default_rng(seed)

    captions: list[str] = []
    engagement: list[float] = []
    topics: list[str] = []
    n_hashtags: list[int] = []
    n_emoji: list[int] = []
    char_len: list[int] = []

    weights = np.array([t.weight for t in TOPICS])

    for _ in range(n):
        topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
        core_text, sentiment = _compose_caption(topic, rng)
        decorated, n_tags, n_emo = _add_surface_noise(core_text, rng)

        noise = rng.normal(0.0, 0.9)
        score = topic.weight + sentiment + noise

        captions.append(decorated)
        engagement.append(float(score))
        topics.append(topic.name)
        n_hashtags.append(n_tags)
        n_emoji.append(n_emo)
        char_len.append(len(decorated))

    # Rescale engagement to a clean positive range (like a normalized
    # engagement-rate proxy) without changing the underlying signal structure.
    arr = np.array(engagement)
    lo = float(arr.min())
    span = float(arr.max() - arr.min()) or 1.0
    scaled = ((arr - lo) / span) * 100.0  # 0..100 engagement index
    engagement = [float(x) for x in scaled]

    return {
        "captions": captions,
        "engagement": engagement,
        "topic": topics,
        "n_hashtags": n_hashtags,
        "n_emoji": n_emoji,
        "char_len": char_len,
    }


def write_csv(path: str, data: dict) -> None:
    """Write a generated dataset to CSV."""
    fields = ["caption", "engagement", "topic", "n_hashtags", "n_emoji", "char_len"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(len(data["captions"])):
            writer.writerow([
                data["captions"][i],
                round(data["engagement"][i], 4),
                data["topic"][i],
                data["n_hashtags"][i],
                data["n_emoji"][i],
                data["char_len"][i],
            ])


def read_csv(path: str) -> dict:
    """Read a dataset CSV back into the dict form generate_dataset returns."""
    data = {
        "captions": [], "engagement": [], "topic": [],
        "n_hashtags": [], "n_emoji": [], "char_len": [],
    }
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["captions"].append(row["caption"])
            data["engagement"].append(float(row["engagement"]))
            data["topic"].append(row["topic"])
            data["n_hashtags"].append(int(row["n_hashtags"]))
            data["n_emoji"].append(int(row["n_emoji"]))
            data["char_len"].append(int(row["char_len"]))
    return data
