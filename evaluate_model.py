
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import preprocess_data


def evaluate_models():
    """Evaluate both trained fraud detection models."""

    BASE_DIR = Path(__file__).resolve().parent

    data_path = BASE_DIR / "Data" / "creditcard.csv"
    model_dir = BASE_DIR / "Models"

    # Prepare the test dataset
    result = preprocess_data(data_path)

    if result is None:
        print("Evaluation stopped: dataset could not be loaded.")
        return

    _, X_test, _, y_test, _ = result

    # Load trained models
    logistic_path = model_dir / "logistic_model.pkl"
    forest_path = model_dir / "random_forest_model.pkl"

    if not logistic_path.exists() or not forest_path.exists():
        print("Trained models not found. Train the models first.")
        return

    models = {
        "Logistic Regression": joblib.load(logistic_path),
        "Random Forest": joblib.load(forest_path)
    }

    results = []

    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        pr_auc = average_precision_score(
            y_test, probabilities
        )

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "PR-AUC": pr_auc
        })

        print(f"\n===== {name} =====")
        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    results_df = pd.DataFrame(results)

    print("\n===== MODEL COMPARISON =====")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        model_dir / "model_comparison.csv",
        index=False
    )

    print("\nComparison saved to models/model_comparison.csv")


if __name__ == "__main__":
    evaluate_models()
