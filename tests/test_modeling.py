import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_text_forensics.modeling import build_svm_pipeline

def test_svm_pipeline_trains_on_small_dataset():
    data = pd.read_csv(ROOT / "data" / "demo_dataset.csv")
    model = build_svm_pipeline()
    model.fit(data["text"], data["label"])
    predictions = model.predict(["This is a simple example sentence."])
    assert len(predictions) == 1
