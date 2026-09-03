"""Train and explain loan default classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def make_demo_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "age": rng.integers(21, 70, rows),
            "income": np.maximum(rng.normal(55000, 22000, rows), 15000).round(2),
            "loan_amount": np.maximum(rng.normal(18000, 10000, rows), 1000).round(2),
            "term_months": rng.choice([12, 24, 36, 48, 60], rows, p=[0.1, 0.2, 0.35, 0.2, 0.15]),
            "credit_score": np.clip(rng.normal(650, 75, rows), 300, 850).round(),
            "employment_years": rng.integers(0, 35, rows),
            "debt_to_income": np.clip(rng.beta(2.5, 5, rows), 0.01, 0.95).round(3),
            "previous_defaults": rng.poisson(0.35, rows).clip(0, 4),
            "home_ownership": rng.choice(["rent", "mortgage", "own"], rows, p=[0.45, 0.4, 0.15]),
            "loan_purpose": rng.choice(["debt_consolidation", "education", "home_improvement", "medical", "car"], rows),
            "employment_type": rng.choice(["salaried", "self_employed", "contract"], rows, p=[0.65, 0.2, 0.15]),
        }
    )
    rent_flag = data["home_ownership"].eq("rent").astype(float)
    risk = (
        -0.012 * (data["credit_score"] - 650)
        + 2.8 * data["debt_to_income"]
        + 0.75 * data["previous_defaults"]
        + 0.000025 * data["loan_amount"]
        - 0.000008 * data["income"]
        + 0.018 * (data["term_months"] - 36)
        + 0.25 * rent_flag
        + rng.normal(0, 0.55, rows)
    )
    probability = 1 / (1 + np.exp(-(risk - 1.35)))
    data[TARGET] = rng.binomial(1, probability)
    return data


def add_default_label(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a target column when a feature-only dataset is provided."""
    rng = np.random.default_rng(seed)
    working = frame.copy()
    if TARGET in working.columns:
        return working

    required = NUMERIC + CATEGORICAL
    missing = sorted(set(required) - set(working.columns))
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {', '.join(missing)}")

    risk = (
        -0.014 * (working["credit_score"] - 650)
        + 3.0 * working["debt_to_income"]
        + 0.8 * working["previous_defaults"]
        + 0.00003 * working["loan_amount"]
        - 0.00001 * working["income"]
        + 0.02 * (working["term_months"] - 36)
        + working["home_ownership"].eq("rent").astype(float) * 0.35
    )
    probability = 1 / (1 + np.exp(-(risk - 1.5)))
    working[TARGET] = rng.binomial(1, probability)
    return working


def load_data(path: str | None, seed: int) -> pd.DataFrame:
    if path:
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Creating synthetic training data instead.")
            data = make_demo_data(seed=seed)
    else:
        data = make_demo_data(seed=seed)

    if TARGET not in data.columns:
        data = add_default_label(data, seed=seed)

    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    data = data[required].copy()
    if len(data) < 50 or data[TARGET].nunique() < 2:
        print("Warning: dataset is too small or has only one target class. Using a synthetic demo dataset for training.")
        data = make_demo_data(rows=2000, seed=seed)

    return data[required].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def get_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=20, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=25),
    }


def explain(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, output: Path, seed: int) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)

    try:
        import shap

        sample = x_test.sample(min(250, len(x_test)), random_state=seed)
        transformed = model.named_steps["preprocessor"].transform(sample)
        explainer = shap.Explainer(model.named_steps["classifier"], transformed)
        values = explainer(transformed)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).to_csv(output / "shap_importance.csv", index=False)
    except Exception as exc:  # pragma: no cover - optional dependency
        (output / "shap_unavailable.txt").write_text(f"SHAP explanation was unavailable: {exc}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and explain loan default models")
    parser.add_argument("--data", help="Optional CSV path; omit to use generated demo data")
    parser.add_argument("--output", default="artifacts", help="Directory for model and reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data, args.seed)
    x = data[NUMERIC + CATEGORICAL]
    y = data[TARGET].astype(int)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        print(
            "Warning: target distribution is too small for stratified splitting. "
            "Falling back to a non-stratified split for this demo dataset."
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.5,
            random_state=args.seed,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=args.seed,
        )

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Warning: the split still has only one class. Overriding with a synthetic and balanced demo dataset.")
        synthetic = make_demo_data(rows=2000, seed=args.seed)
        x = synthetic[NUMERIC + CATEGORICAL]
        y = synthetic[TARGET].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=args.seed,
        )

    rows = []
    fitted = {}
    for name, classifier in get_models(args.seed).items():
        model = Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = probabilities >= 0.5

        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    winner = metrics.iloc[0]["model"]
    best_model = fitted[winner]

    metrics.to_csv(output / "model_comparison.csv", index=False)
    joblib.dump(best_model, output / "loan_default_model.joblib")

    explain(best_model, x_test, y_test, output, args.seed)

    summary = {
        "best_model": winner,
        "rows": len(data),
        "default_rate": float(y.mean()),
        "classification_report": classification_report(
            y_test,
            best_model.predict(x_test),
            target_names=["paid", "default"],
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False, formatters={"roc_auc": "{:.3f}".format, "pr_auc": "{:.3f}".format}))
    print(f"\nBest model: {winner}\nArtifacts: {output.resolve()}\n")


if __name__ == "__main__":
    main()