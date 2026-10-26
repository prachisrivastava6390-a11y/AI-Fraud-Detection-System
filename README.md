# AI Fraud Detection System

A Streamlit app that scores transactions for fraud using a trained scikit-learn model. Upload a transaction CSV, get a fraud prediction and probability for every row, and download the results.

**Live app:** [TODO: Streamlit Cloud link]

## What it does

- Takes a CSV of transactions as input
- Runs a trained scikit-learn model (loaded with joblib) on every row
- Adds three columns to the output:
  - `Fraud_Prediction`: 1 = model flags the transaction as fraud, 0 = not flagged
  - `Fraud_Probability`: model confidence that the transaction is fraud (0-100%)
  - `Prediction_Result`: text label (e.g. "Potential Fraud")
- Shows charts (Altair) and lets you download the scored file as `fraud_detection_results.csv`

## Tech stack

Python, Streamlit, pandas, scikit-learn, joblib, Altair

## Run locally

```bash
git clone https://github.com/prachisrivastava6390-a11y/AI-Fraud-Detection-System.git
cd AI-Fraud-Detection-System
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Note: this project was built with Python 3.11+ (pandas 3.x needs it).

## Model

- Algorithm: [TODO: e.g. Random Forest / Logistic Regression]
- Dataset: [TODO: dataset name and link; not included in the repo because of file size]
- Evaluation: [TODO: precision / recall / F1 on the test set]

## Limitations

- **The model misses some real frauds.** Rows where the true label is fraud but the model predicted 0 (false negatives) are not flagged, so recall is below 100%. Filtering the output on "Potential Fraud" only shows what the model caught, not what it missed. To audit misses, filter `Class = 1` and look for `Fraud_Prediction = 0`.
- Some flagged transactions have low confidence (probabilities close to 50-60%), so they need manual review rather than automatic action.
- Fraud data is highly imbalanced, so accuracy alone is misleading. Judge the model on precision and recall.
- This is a portfolio/demo project trained on a public dataset. It is not production-ready and has not been validated on real transaction data.

## Author

Prachi, [GitHub](https://github.com/prachisrivastava6390-a11y)
