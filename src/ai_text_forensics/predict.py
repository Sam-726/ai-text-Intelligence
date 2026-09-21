from __future__ import annotations

from typing import Any

import numpy as np


def predict_text(model, text: str) -> dict[str, Any]:
    label = str(model.predict([text])[0])
    result: dict[str, Any] = {"label": label}
    if hasattr(model, "decision_function"):
        result["decision_score"] = float(np.asarray(model.decision_function([text])).ravel()[0])
        result["confidence_note"] = (
            "The decision score is model evidence, not a calibrated probability."
        )
    return result
