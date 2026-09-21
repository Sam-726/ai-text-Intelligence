from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_text_forensics.modeling import load_model
from ai_text_forensics.predict import predict_text

MODEL_PATH = ROOT / "artifacts" / "svm_pipeline.joblib"

st.set_page_config(page_title="AI Text Forensics", page_icon="🧪")
st.title("🧪 AI Text Forensics")
st.caption("Master's-level demonstration of interpretable text classification.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run python scripts/train.py first.")
    st.stop()

model = load_model(MODEL_PATH)
text = st.text_area("Enter text to analyze", height=220, placeholder="Paste a paragraph here...")

if st.button("Analyze text", type="primary"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        result = predict_text(model, text)
        st.subheader("Result")
        st.metric("Predicted class", result["label"])
        if "decision_score" in result:
            st.write(f"Decision score: {result['decision_score']:.3f}")
            st.info(result["confidence_note"])

st.divider()
st.markdown("**Important:** This is a learning/portfolio prototype. A small binary classifier cannot reliably determine authorship or the exact LLM source of arbitrary text.")
