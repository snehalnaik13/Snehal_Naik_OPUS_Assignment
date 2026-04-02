"""
ML-based NLP scoring using TF-IDF + Random Forest

Pipeline:
Text → TF-IDF → Random Forest → Risk Score
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


# ── Paths ────────────────────────────────────────────────────────────────────
VECTORIZER_PATH = "models/tfidf.pkl"
MODEL_PATH = "models/nlp_rf_model.pkl"


# ── Train NLP Model ──────────────────────────────────────────────────────────
def train_nlp_model(
    df: pd.DataFrame,
    text_col: str = "Description",
    target_col: str = "ChargebackRate"
):
    """
    Train Random Forest using TF-IDF features

    WHY ChargebackRate as target?
    → It reflects actual financial risk behaviour
    """

    texts = df[text_col].fillna("").astype(str)

    # Step 1: TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    # Step 2: Target
    y = df[target_col].values

    # Step 3: Model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save artifacts
    os.makedirs("../models", exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("[NLP-ML] Model trained and saved")

    return vectorizer, model


# ── Load Model ───────────────────────────────────────────────────────────────
def load_nlp_model():
    """
    Load trained TF-IDF vectorizer and Random Forest model
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

def compute_nlp_risk_score(description: str) -> float:
    """
    Streamlit-safe wrapper function.
    Loads vectorizer + model internally.
    """
    vectorizer, model = load_nlp_model()
    return compute_nlp_risk_score_ml(description, vectorizer, model)


# ── Compute NLP Risk Score ───────────────────────────────────────────────────
def compute_nlp_risk_score_ml(
    description: str,
    vectorizer,
    model
) -> float:
    """
    Predict risk score using trained Random Forest model
    """

    X = vectorizer.transform([str(description)])

    score = model.predict(X)[0]

    # Normalize to 0–1
    score = np.clip(score, 0.0, 1.0)

    return round(float(score), 4)


# ── Apply to DataFrame ───────────────────────────────────────────────────────
def add_nlp_scores(
    df: pd.DataFrame,
    text_col: str = "Description",
    train: bool = False
) -> pd.DataFrame:
    """
    Add nlp_risk_score column

    train=True  → trains model
    train=False → loads model
    """

    if train:
        vectorizer, model = train_nlp_model(df, text_col=text_col)
    else:
        vectorizer, model = load_nlp_model()

    texts = df[text_col].fillna("").astype(str)
    X = vectorizer.transform(texts)

    scores = model.predict(X)

    # Min–Max normalization
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    # Clip and assign
    df["nlp_risk_score"] = np.clip(scores, 0.0, 1.0).round(4)

    print(
        f"[NLP-ML] Scores computed | "
        f"Mean: {df['nlp_risk_score'].mean():.3f} | "
        f"Max: {df['nlp_risk_score'].max():.3f}"
    )

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")

    df = add_nlp_scores(df, train=True)

    df[["MerchantID", "Description", "nlp_risk_score"]].head(10).to_csv(
        "data/nlp_ml_scores_sample.csv",
        index=False
    )

    print("[NLP-ML] Sample saved")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_PATH = "data/transactions.csv"
    OUTPUT_PATH = "data/transactions.csv"

    # Load feature-engineered data
    df = pd.read_csv(INPUT_PATH)

    print(f"[NLP-ML] Loaded data → {INPUT_PATH}")
    print(f"[NLP-ML] Rows: {len(df)}")

    # Add NLP risk scores
    df = add_nlp_scores(df, text_col="Description", train=True)

    # Save NLP-scored dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[NLP-ML] Saved → {OUTPUT_PATH}")
    print(df[["MerchantID", "Description", "nlp_risk_score"]].head())