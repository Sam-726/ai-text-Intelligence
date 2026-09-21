import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_text_forensics.features import FEATURE_COLUMNS, extract_stylometric_features

def test_feature_shape_and_values():
    frame = extract_stylometric_features(["Hello world!", "One, two?"])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 2
    assert frame.loc[0, "word_count"] == 2
    assert frame.loc[1, "question_count"] == 1
