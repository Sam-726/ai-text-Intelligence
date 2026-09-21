from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_text_forensics.modeling import build_svm_pipeline, evaluate_classifier, save_model

DATA_PATH = ROOT / "data" / "demo_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "svm_pipeline.joblib"

def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"])
    model = build_svm_pipeline()
    model.fit(x_train, y_train)
    result = evaluate_classifier(model, x_test, y_test)

    print("=== AI Text Forensics: SVM baseline ===")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"ROC-AUC:  {result.roc_auc:.3f}" if result.roc_auc is not None else "ROC-AUC: n/a")
    print("\nClassification report:")
    print(result.report)
    print("Confusion matrix:")
    for row in result.confusion_matrix:
        print(row)

    save_model(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
