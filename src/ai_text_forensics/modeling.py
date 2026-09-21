from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .features import extract_stylometric_features


def _stylometry(texts) -> pd.DataFrame:
    return extract_stylometric_features(texts)


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    roc_auc: float | None
    report: str
    confusion_matrix: list[list[int]]


def build_svm_pipeline() -> Pipeline:
    feature_union = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=3000,
                ),
            ),
            (
                "stylometry",
                Pipeline(
                    [
                        ("extract", FunctionTransformer(_stylometry, validate=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", feature_union),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def evaluate_classifier(model: Pipeline, x_test, y_test) -> EvaluationResult:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, digits=3, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()
    accuracy = float(accuracy_score(y_test, predictions))
    roc_auc = None
    if len(set(y_test)) == 2:
        scores = model.decision_function(x_test)
        roc_auc = float(roc_auc_score((y_test == "ai").astype(int), scores))
    return EvaluationResult(accuracy, roc_auc, report, matrix)


def save_model(model: Pipeline, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
