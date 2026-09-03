"""Predict loan default risk using a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

TARGET = "loan_default"
NUMERIC = [
    "age",
    "income",
    "loan_amount",
    "term_months",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "previous_defaults",
]
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_type"]


def prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = NUMERIC + CATEGORICAL
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
    if TARGET in frame.columns:
        return frame[required].copy()
    return frame[required].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict loan default risk")
    parser.add_argument("--model", required=True, help="Path to saved trained model (.joblib)")
    parser.add_argument("--input", required=True, help="CSV file containing borrower features")
    parser.add_argument("--output", default="predictions.csv", help="CSV path to store predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    features = prepare_input(df)
    model = joblib.load(args.model)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = pd.DataFrame(
        {
            "probability_default": probabilities,
            "predicted_default": (probabilities >= 0.5).astype(int),
        }
    )

    if TARGET in df.columns:
        predictions[TARGET] = df[TARGET].astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    print(predictions.head().to_string(index=False))
    print(f"\nSaved predictions to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
