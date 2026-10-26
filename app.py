import logging

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_app")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

st.title("💳 AI-Powered Fraud Detection System")
st.write(
    "Analyze credit card transactions using Machine Learning "
    "to identify potentially fraudulent activity."
)

# ============================================================
# 2. MODEL FILE PATHS
# (Folder name must match exactly: "Models" on Linux is case-sensitive)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"

LOGISTIC_PATH = MODEL_DIR / "logistic_model.pkl"
FOREST_PATH = MODEL_DIR / "random_forest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

MAX_TABLE_ROWS = 500  # rows shown on screen; download has everything

REQUIRED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

# ============================================================
# 3. CHECK + LOAD MODELS
# ============================================================

missing_model_files = [
    p.name for p in (LOGISTIC_PATH, FOREST_PATH, SCALER_PATH) if not p.exists()
]

if missing_model_files:
    st.error(
        f"Missing model files in the Models folder: {missing_model_files}. "
        "Run train_model.py first."
    )
    st.stop()


@st.cache_resource
def load_models():
    """Load trained models and scaler (trusted local files only)."""
    return (
        joblib.load(LOGISTIC_PATH),
        joblib.load(FOREST_PATH),
        joblib.load(SCALER_PATH),
    )


try:
    logistic_model, random_forest_model, scaler = load_models()
except Exception:
    logger.exception("Model loading failed")
    st.error(
        "The models could not be loaded. Check the files inside the Models "
        "folder and that scikit-learn version matches the one used for training."
    )
    st.stop()

st.success("Trained models loaded successfully.")

# ============================================================
# 4. MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()
st.header("📈 Model Performance Comparison")

st.write(
    "Compare Precision, Recall, F1 Score, and PR-AUC. "
    "Accuracy alone is not sufficient for an imbalanced "
    "fraud detection dataset."
)

if COMPARISON_PATH.exists():
    try:
        comparison_df = pd.read_csv(COMPARISON_PATH)

        if {"Model", "F1 Score"}.issubset(comparison_df.columns):
            st.dataframe(
                comparison_df.round(4),
                width="stretch",
                hide_index=True,
            )

            st.subheader("F1 Score Comparison")

            # Fixed 0-1 axis so low scores are never hidden.
            f1_chart = (
                alt.Chart(comparison_df)
                .mark_bar()
                .encode(
                    x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("F1 Score:Q", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Model", alt.Tooltip("F1 Score:Q", format=".4f")],
                )
            )
            st.altair_chart(f1_chart, width="stretch")

            valid_f1 = comparison_df.dropna(subset=["F1 Score"])
            if not valid_f1.empty:
                best_model = valid_f1.loc[valid_f1["F1 Score"].idxmax()]
                st.success(
                    f"Best model by F1 Score: {best_model['Model']} "
                    f"({best_model['F1 Score']:.4f})"
                )
        else:
            st.warning(
                "The comparison CSV is missing the required 'Model' or "
                "'F1 Score' column. Run evaluate_model.py again."
            )
    except Exception:
        logger.exception("Could not read comparison CSV")
        st.warning(
            "The model comparison file could not be read. "
            "Run evaluate_model.py again."
        )
else:
    st.warning(
        "Model comparison results are unavailable. Run evaluate_model.py first."
    )

# ============================================================
# 5. TRANSACTION CSV UPLOAD
# ============================================================

st.divider()
st.header("📂 Transaction Analysis")

st.write(
    "Upload a CSV containing Time, V1–V28, and Amount. "
    "If the CSV also contains a valid Class column "
    "(0 = legitimate, 1 = fraud), evaluation metrics can be calculated."
)

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a transaction CSV file to begin the analysis.")
    st.stop()

# Clear previous results when the uploaded file changes.
upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"

if st.session_state.get("current_upload_signature") != upload_signature:
    st.session_state["current_upload_signature"] = upload_signature
    for key in ("fraud_results", "fraud_model_name", "uploaded_evaluation"):
        st.session_state.pop(key, None)

# ============================================================
# 6. READ AND PREVIEW UPLOADED DATA
# ============================================================


@st.cache_data(show_spinner="Reading CSV...")
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(file_bytes))


try:
    transaction_data = read_csv_cached(uploaded_file.getvalue())
except Exception:
    logger.exception("CSV read failed")
    st.error(
        "The uploaded file could not be read as a CSV. "
        "Please check the file and try again."
    )
    st.stop()

if transaction_data.empty:
    st.error("The uploaded CSV contains no transaction rows.")
    st.stop()

has_actual_labels = "Class" in transaction_data.columns

st.subheader("Uploaded Transaction Data")
st.write("Dataset shape:", transaction_data.shape)
st.dataframe(transaction_data.head(), width="stretch")

# ============================================================
# 7. VALIDATE REQUIRED FEATURES
# ============================================================

st.divider()
st.subheader("Data Validation")

missing_columns = [c for c in REQUIRED_COLUMNS if c not in transaction_data.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.info(
        "The model expects Time, V1 through V28, and Amount. "
        "Check the column names in your CSV."
    )
    st.stop()

st.success("All required feature columns are present.")

# Same feature order as training.
feature_data = transaction_data[REQUIRED_COLUMNS].apply(
    pd.to_numeric, errors="coerce"
)

if feature_data.isna().any().any():
    invalid_columns = feature_data.columns[feature_data.isna().any()].tolist()
    st.error(
        "Missing or non-numeric values were found in: "
        f"{invalid_columns}. Clean the CSV and upload again."
    )
    st.stop()

if feature_data.isin([float("inf"), float("-inf")]).any().any():
    st.error("The CSV contains infinite values. Correct them before analysis.")
    st.stop()

if (feature_data["Amount"] < 0).any():
    st.error("Transaction Amount cannot be negative. Please check your data.")
    st.stop()

# ============================================================
# 8. VALIDATE ACTUAL CLASS LABELS
# ============================================================

actual_labels = None
labels_are_valid = False

if has_actual_labels:
    actual_labels = pd.to_numeric(transaction_data["Class"], errors="coerce")

    if actual_labels.isna().any() or not actual_labels.isin([0, 1]).all():
        st.error(
            "The Class column must contain only 0 (legitimate) or 1 (fraud), "
            "with no missing or non-numeric values."
        )
        st.stop()

    actual_labels = actual_labels.astype(int)
    labels_are_valid = True

st.success("Transaction values passed validation.")

if labels_are_valid:
    if actual_labels.nunique() == 2:
        st.caption(
            "Both classes are present. Evaluation metrics can be calculated "
            "after prediction."
        )
    else:
        st.warning(
            "The Class column contains only one class. Predictions can still "
            "be generated, but useful evaluation metrics require both classes."
        )
else:
    st.caption(
        "No Class column was found. Predictions are possible, but evaluation "
        "metrics cannot be calculated without actual labels."
    )

# ============================================================
# 9. PREDICTION SETTINGS
# ============================================================

st.divider()
st.subheader("⚙️ Prediction Settings")

selected_model_name = st.selectbox(
    "Choose a Machine Learning model",
    ["Random Forest", "Logistic Regression"],
)

threshold = st.slider(
    "Fraud probability threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "Lower threshold = more transactions flagged (higher recall, lower "
        "precision). Higher threshold = fewer flags (higher precision, lower "
        "recall)."
    ),
)

st.write(
    f"Selected model: **{selected_model_name}** | "
    f"Threshold: **{threshold:.2f}**"
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}

# ============================================================
# 10. PREDICT TRANSACTIONS
# ============================================================

if st.button("Analyze Transactions", type="primary"):
    try:
        with st.spinner("Analyzing transactions..."):
            scaled_features = scaler.transform(feature_data)

            # Probabilities computed once per model and reused.
            probabilities_by_model = {
                name: model.predict_proba(scaled_features)[:, 1]
                for name, model in models.items()
            }

            selected_probs = probabilities_by_model[selected_model_name]
            predictions = (selected_probs >= threshold).astype(int)

            results = transaction_data.copy()
            results["Fraud_Prediction"] = predictions
            results["Fraud_Probability"] = (selected_probs * 100).round(2)
            results["Prediction_Result"] = results["Fraud_Prediction"].map(
                {0: "Legitimate", 1: "Potential Fraud"}
            )

            st.session_state["fraud_results"] = results
            st.session_state["fraud_model_name"] = selected_model_name
            st.session_state["fraud_threshold"] = threshold
            st.session_state.pop("uploaded_evaluation", None)

            if labels_are_valid and actual_labels.nunique() == 2:
                rows = []
                for name, probs in probabilities_by_model.items():
                    preds = (probs >= threshold).astype(int)
                    rows.append(
                        {
                            "Model": name,
                            "Precision": precision_score(
                                actual_labels, preds, zero_division=0
                            ),
                            "Recall": recall_score(
                                actual_labels, preds, zero_division=0
                            ),
                            "F1 Score": f1_score(
                                actual_labels, preds, zero_division=0
                            ),
                            "PR-AUC": average_precision_score(
                                actual_labels, probs
                            ),
                        }
                    )
                st.session_state["uploaded_evaluation"] = pd.DataFrame(rows)

    except Exception:
        logger.exception("Prediction failed")
        st.error(
            "Prediction failed. Check that the uploaded columns and "
            "preprocessing match the trained models."
        )

# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

if "fraud_results" in st.session_state:
    results = st.session_state["fraud_results"]
    result_model_name = st.session_state.get("fraud_model_name", "Selected model")
    result_threshold = st.session_state.get("fraud_threshold", 0.5)

    st.divider()
    st.header("📊 Fraud Detection Results")
    st.caption(
        f"Predictions generated using: {result_model_name} "
        f"(threshold {result_threshold:.2f})"
    )

    total_transactions = len(results)
    fraud_count = int((results["Fraud_Prediction"] == 1).sum())
    legitimate_count = total_transactions - fraud_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Potential Fraud", f"{fraud_count:,}")
    col3.metric("Legitimate Transactions", f"{legitimate_count:,}")

    st.subheader("Prediction Summary")
    summary = (
        results["Prediction_Result"]
        .value_counts()
        .rename_axis("Transaction Status")
        .reset_index(name="Count")
    )
    st.bar_chart(summary.set_index("Transaction Status"))

    # Show only flagged rows, highest risk first (keeps the UI fast).
    st.subheader("Top Flagged Transactions")
    flagged = results[results["Fraud_Prediction"] == 1].sort_values(
        "Fraud_Probability", ascending=False
    )

    if flagged.empty:
        st.info("No transactions were flagged at this threshold.")
    else:
        if len(flagged) > MAX_TABLE_ROWS:
            st.caption(
                f"Showing the top {MAX_TABLE_ROWS:,} of {len(flagged):,} "
                "flagged transactions. Download the CSV for the full list."
            )
        st.dataframe(
            flagged.head(MAX_TABLE_ROWS),
            width="stretch",
            hide_index=True,
        )

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Prediction Results",
        data=csv_data,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
    )

    if "uploaded_evaluation" in st.session_state:
        st.divider()
        st.subheader("📋 Evaluation Metrics for Uploaded Dataset")
        st.write(
            "These metrics compare both models with the actual values in the "
            "uploaded Class column at the selected threshold (PR-AUC is "
            "threshold-independent). They describe performance on this "
            "uploaded dataset and may not represent performance on unseen "
            "transactions."
        )
        st.dataframe(
            st.session_state["uploaded_evaluation"].round(4),
            width="stretch",
            hide_index=True,
        )

    st.info(
        "These predictions are model-generated risk indicators, not proof "
        "that a transaction is fraudulent. Review flagged transactions before "
        "taking action."
    )