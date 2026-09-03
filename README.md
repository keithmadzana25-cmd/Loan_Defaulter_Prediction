# Loan Default Prediction with Explainable AI

## Project title
Loan Default Prediction with Explainable AI

## Objective
This project builds a simple but effective machine learning system that predicts whether a borrower is likely to default on a loan. The system compares several classification algorithms, selects the best model using validation metrics, and explains the predictions using feature importance and SHAP-based insight when available.

## Business problem
Banks and lenders need to identify risky borrowers early. A strong default prediction model helps reduce credit risk and supports faster, more responsible lending decisions.

## Dataset
The project supports two cases:

1. A real loan dataset with a target column named loan_default
2. A feature-only CSV without the label, in which case the system automatically creates a synthetic default target for demonstration and model testing

The example dataset in the data folder is a feature-only loan file. The training pipeline automatically adds a loan_default label so the project can run without manual preparation.

## Features used
Numeric features:
- age
- income
- loan_amount
- term_months
- credit_score
- employment_years
- debt_to_income
- previous_defaults

Categorical features:
- home_ownership
- loan_purpose
- employment_type

Target:
- loan_default

## Algorithms compared
The training pipeline evaluates multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- K-Nearest Neighbors

## Libraries used
- Python
- pandas
- NumPy
- scikit-learn
- joblib
- SHAP
- matplotlib

## Project structure
- README.md — main project overview and usage guide
- docs/PROJECT_REPORT.md — detailed project report
- requirements.txt — Python dependencies
- data/example_loan.csv — sample loan dataset
- src/train.py — model training and evaluation pipeline
- src/predict.py — prediction script for new borrower data
- artifacts/ — generated model and evaluation files after training

## How to run
### Option 1: Python command-line workflow
1. Create a virtual environment (optional but recommended):
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python src/train.py --data data/example_loan.csv --output artifacts

4. Predict with the trained model:
   python src/predict.py --model artifacts/loan_default_model.joblib --input data/example_loan.csv --output predictions.csv

### Option 2: User-friendly web interface
1. Install dependencies:
   pip install -r requirements.txt

2. Start the app:
   streamlit run app.py

3. Open the browser link shown in the terminal, then enter borrower information and click Predict.

The web app loads the trained model from the artifacts folder and gives a default-risk score, class prediction, and a plain-language risk summary.

5. Open the detailed report:
   docs/PROJECT_REPORT.md

## Output files
After training, the system creates these artifacts in the output folder:
- model_comparison.csv — comparison of all tested models
- loan_default_model.joblib — saved best-performing trained model
- feature_importance.csv — permutation-based feature importance
- shap_importance.csv — SHAP-based importance (when SHAP is available)
- summary.json — summary of model selection and evaluation

## Model evaluation metrics
The project uses:
- ROC-AUC
- Precision-Recall AUC
- Accuracy
- Precision
- Recall
- F1-score

These metrics are used to select the best model for deployment.

## Explainable AI
The project makes model decisions more understandable by reporting:
- feature importance from permutation importance
- SHAP importance values when the library is available

This supports transparency in credit decisions.

## Example expected workflow
- Load borrower information
- Run preprocessing on numeric and categorical variables
- Train multiple models on the data
- Compare results
- Save the best model
- Use the model to predict default risk on new applicants

## Notes
This project is intentionally kept simple so it is easy to understand and extend. It is suitable for academic projects, demos, and early-stage credit risk modeling.

## Result
This system answers the project title by creating a practical loan default prediction framework with model comparison, explainability, and deployable prediction logic.

## Quick project summary
- Goal: predict whether a borrower may default on a loan
- Best model: selected automatically from several classifiers
- Explainability: permutation importance and SHAP output
- Deployment: model saved as a joblib artifact for future predictions
- Documentation: README and project report included
