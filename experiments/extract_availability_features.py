"""Extract simple availability-heuristic-inspired features from essay text.

This script reads enriched prediction exports for validation and test splits,
derives explainable proxy features from essay text, and writes one combined CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


VAL_PATH = Path("results_asap_debug/test_eval_results/val_predictions_with_ids.csv")
TEST_PATH = Path("results_asap_debug/test_eval_results/test_predictions_with_ids.csv")
OUT_PATH = Path("experiments/outputs/availability_features.csv")


TOKEN_RE = re.compile(r"[a-zA-Z']+")


# Concrete words are tangible objects, sensory descriptors, and physical places.
CONCRETE_WORDS = {
    "book", "books", "desk", "classroom", "teacher", "student", "school",
    "home", "house", "room", "street", "car", "bus", "train", "phone",
    "computer", "food", "water", "money", "paper", "pencil", "pen",
    "dog", "cat", "tree", "park", "city", "town", "road", "door", "window",
    "red", "blue", "green", "black", "white", "loud", "quiet", "hot", "cold",
    "smell", "taste", "touch", "sound", "light", "dark",
}

# Specific detail words include numbers, named entities, and discourse markers
# that usually appear in concrete examples and detailed evidence.
SPECIFIC_MARKERS = {
    "for example", "for instance", "specifically", "in particular",
    "such as", "according to", "evidence", "data", "percent", "because",
}

# Emotional salience proxies: affect words and punctuation intensity.
EMOTION_WORDS = {
    "love", "hate", "happy", "sad", "angry", "afraid", "fear", "excited",
    "worried", "anxious", "proud", "ashamed", "frustrated", "joy", "pain",
    "upset", "surprised", "disappointed", "hope", "hopeless",
}

# Personal experience salience proxies: first-person perspective + memory/time anchors.
PERSONAL_WORDS = {
    "i", "me", "my", "mine", "myself", "we", "our", "ours", "us",
    "remember", "experienced", "experience", "felt", "when", "once", "last",
}

# Narrative events proxies: event verbs and sequencing words.
EVENT_WORDS = {
    "went", "saw", "met", "told", "said", "asked", "found", "lost", "won",
    "happened", "started", "stopped", "arrived", "left", "called", "walked",
    "ran", "drove", "then", "next", "after", "before", "finally",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def safe_rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def concreteness(text: str) -> float:
    """Approximate concreteness by fraction of concrete/sensory terms."""
    tokens = tokenize(text)
    count = sum(1 for t in tokens if t in CONCRETE_WORDS)
    return safe_rate(count, len(tokens))


def specificity(text: str) -> float:
    """Approximate specificity using detail markers + numeric token density."""
    lower = text.lower()
    marker_hits = sum(1 for m in SPECIFIC_MARKERS if m in lower)
    number_hits = len(re.findall(r"\b\d+(?:\.\d+)?\b", lower))
    token_count = len(tokenize(lower))
    return safe_rate(marker_hits + number_hits, token_count)


def emotional_salience(text: str) -> float:
    """Approximate emotional salience from affect vocabulary and '!' emphasis."""
    tokens = tokenize(text)
    emotion_hits = sum(1 for t in tokens if t in EMOTION_WORDS)
    exclam_hits = text.count("!")
    return safe_rate(emotion_hits + exclam_hits, max(len(tokens), 1))


def personal_experience_salience(text: str) -> float:
    """Approximate personal experience via first-person and memory cues."""
    tokens = tokenize(text)
    personal_hits = sum(1 for t in tokens if t in PERSONAL_WORDS)
    return safe_rate(personal_hits, len(tokens))


def narrative_event_density(text: str) -> float:
    """Approximate event density via event verbs and temporal sequencing terms."""
    tokens = tokenize(text)
    event_hits = sum(1 for t in tokens if t in EVENT_WORDS)
    return safe_rate(event_hits, len(tokens))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["text"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "essay_id": df["essay_id"].values,
            "split": df["split"].values,
            "concreteness": [concreteness(t) for t in texts],
            "specificity": [specificity(t) for t in texts],
            "emotional_salience": [emotional_salience(t) for t in texts],
            "personal_experience_salience": [personal_experience_salience(t) for t in texts],
            "narrative_event_density": [narrative_event_density(t) for t in texts],
        }
    )
    return out


def main() -> None:
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required = {"essay_id", "split", "text"}
    for name, df in [("val", val_df), ("test", test_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} file missing required columns: {sorted(missing)}")

    combined = pd.concat([val_df, test_df], axis=0, ignore_index=True)
    feature_df = extract_features(combined)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(feature_df)} rows to {OUT_PATH}")
    print("Columns:", list(feature_df.columns))
    print("Split counts:", feature_df["split"].value_counts().to_dict())
    print("Any NaN:", bool(feature_df.isna().any().any()))
    print("Feature ranges:")
    for c in [
        "concreteness",
        "specificity",
        "emotional_salience",
        "personal_experience_salience",
        "narrative_event_density",
    ]:
        print(f"  {c}: min={feature_df[c].min():.6f}, max={feature_df[c].max():.6f}")


if __name__ == "__main__":
    main()
