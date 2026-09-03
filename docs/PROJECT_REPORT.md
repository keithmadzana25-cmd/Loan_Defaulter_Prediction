# Project Report

## 1. Project overview
This project develops a loan default prediction system that helps lenders identify borrowers with a higher probability of default. The model is designed to be simple, explainable, and easy to run on a local machine.

## 2. Problem statement
Loan default risk is a major issue in credit decision-making. A lender needs to estimate whether a borrower will fail to repay a loan. A robust predictive model reduces financial risk and supports better lending decisions.

## 3. Objectives
- Build a reliable default prediction model
- Compare several machine learning algorithms
- Save the best-performing model
- Explain prediction outcomes with feature importance
- Provide a streamlined project workflow

## 4. Data and features
The project uses loan-related fields such as income, amount, debt-to-income ratio, credit score, employment type, and loan purpose. The dataset is passed through a preprocessing pipeline that handles both numeric and categorical variables.

### Numeric features
- age
- income
- loan_amount
- term_months
- credit_score
- employment_years
- debt_to_income
- previous_defaults

### Categorical features
- home_ownership
- loan_purpose
- employment_type

### Target variable
- loan_default

## 5. Methodology
The project follows a simple machine learning pipeline:
1. Load loan data
2. Check if the target exists
3. Generate a synthetic label for feature-only demo datasets when needed
4. Split data into training and testing sets
5. Run preprocessing for numeric and categorical variables
6. Train multiple models
7. Compare performance using evaluation metrics
8. Select the best model
9. Save model and feature-importance outputs

## 6. Algorithms used
The training script evaluates the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- K-Nearest Neighbors

## 7. Evaluation metrics
The project measures model quality with:
- ROC-AUC
- Precision-Recall AUC
- Accuracy
- Precision
- Recall
- F1-score

## 8. Explainable AI
To improve interpretability, the system includes:
- permutation importance
- SHAP-based feature importance when SHAP is available

This helps identify which borrower factors most strongly influence the default prediction.

## 9. File structure
- README.md — main project guide
- requirements.txt — environment dependencies
- data/example_loan.csv — sample loan dataset
- src/train.py — training logic
- src/predict.py — prediction pipeline
- artifacts/ — model and evaluation outputs
- docs/PROJECT_REPORT.md — project report

## 10. Results and outcome
The final system is operational and can train a model, compare classifiers, explain the outcome, and make predictions on new loan applications. It is suitable for academic work, demos, and early-stage credit-risk analysis.

## 11. Conclusion
This project demonstrates a practical and understandable machine learning solution for loan default prediction. It balances performance, simplicity, and transparency, which is valuable for both technical and non-technical users.
