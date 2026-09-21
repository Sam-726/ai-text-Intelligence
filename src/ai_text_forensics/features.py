from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "type_token_ratio",
    "exclamation_count",
    "question_count",
    "comma_count",
    "digit_count",
]


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]


def extract_stylometric_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        words = _words(text)
        sentences = _sentences(text)
        word_count = len(words)
        sentence_count = len(sentences)
        rows.append(
            {
                "char_count": float(len(text)),
                "word_count": float(word_count),
                "sentence_count": float(sentence_count),
                "avg_word_length": float(
                    np.mean([len(w) for w in words]) if words else 0.0
                ),
                "avg_sentence_length": float(
                    word_count / sentence_count if sentence_count else 0.0
                ),
                "type_token_ratio": float(
                    len(set(words)) / word_count if word_count else 0.0
                ),
                "exclamation_count": float(text.count("!")),
                "question_count": float(text.count("?")),
                "comma_count": float(text.count(",")),
                "digit_count": float(sum(c.isdigit() for c in text)),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
