import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/loan_default_model.joblib")
FEATURES = [
    "age",
    "income",
    "loan_amount",
    "term_months",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "previous_defaults",
    "home_ownership",
    "loan_purpose",
    "employment_type",
]

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="centered")


def load_model() -> object:
    if not MODEL_PATH.exists():
        st.error("Model not found. Train the model first: python src/train.py --data data/example_loan.csv --output artifacts")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_summary():
    summary_path = Path("artifacts/summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"best_model": "not available"}


st.title("Loan Default Prediction Dashboard")
st.caption("A user-friendly loan risk prediction app using explainable machine learning.")

summary = load_feature_summary()
if summary.get("best_model") != "not available":
    st.info(f"Current best model: {summary['best_model']}")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        income = st.number_input("Annual income", min_value=1000, max_value=500000, value=60000, step=1000)
        loan_amount = st.number_input("Loan amount", min_value=1000, max_value=500000, value=20000, step=1000)
        term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
        credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=680)
        employment_years = st.number_input("Employment years", min_value=0, max_value=50, value=6)

    with col2:
        debt_to_income = st.slider("Debt-to-income ratio", min_value=0.0, max_value=1.0, value=0.32, step=0.01)
        previous_defaults = st.number_input("Previous defaults", min_value=0, max_value=10, value=0)
        home_ownership = st.selectbox("Home ownership", ["rent", "mortgage", "own"])
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["debt_consolidation", "education", "home_improvement", "medical", "car"],
            index=0,
        )
        employment_type = st.selectbox("Employment type", ["salaried", "self_employed", "contract"])

    submitted = st.form_submit_button("Predict default risk")

if submitted:
    model = load_model()
    input_data = pd.DataFrame(
        [{
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "previous_defaults": previous_defaults,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "employment_type": employment_type,
        }]
    )

    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.5)
    risk_label = "High Risk" if prediction == 1 else "Low Risk"

    st.subheader("Prediction result")
    st.metric("Probability of default", f"{probability * 100:.2f}%")
    st.metric("Risk classification", risk_label)

    if probability >= 0.7:
        risk_message = "This applicant has a high chance of default. The loan should be reviewed carefully or declined if policy requires stricter screening."
    elif probability >= 0.4:
        risk_message = "This applicant shows moderate risk. A review of income stability and repayment capacity is recommended."
    else:
        risk_message = "This applicant appears to have relatively low default risk based on the current model."

    st.write(risk_message)

    st.subheader("Input summary")
    st.dataframe(input_data, use_container_width=True)

    st.caption("Model feature set used: " + ", ".join(FEATURES))
