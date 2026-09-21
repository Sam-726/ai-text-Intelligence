# AI Text Forensics

A Master's-level NLP/ML project for detecting whether text is human-written or AI-generated.

## Current version
The first milestone contains a reproducible SVM baseline using TF-IDF plus interpretable stylometric features, evaluation metrics, tests, a small demonstration dataset, and a Streamlit interface.

This is an educational prototype, not a production-grade AI detector. Performance depends on the training data and can degrade on unseen domains, models, paraphrases, or languages.

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train
```bash
python scripts/train.py
```

## Run the app
```bash
streamlit run app.py
```

## Planned milestones
- Add transformer embeddings.
- Add multiclass LLM attribution.
- Add a simple authorship/style similarity experiment.
