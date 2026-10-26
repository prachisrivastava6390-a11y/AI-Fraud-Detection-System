
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_data


def train_models():
    """Train and save two fraud detection models."""

    data_path = Path("Data") / "creditcard.csv"
    model_dir = Path(__file__).resolve().parent / "Models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the dataset
    result = preprocess_data(data_path)

    if result is None:
        print("Training stopped: dataset could not be loaded.")
        return

    X_train, X_test, y_train, y_test, scaler = result

    # Model 1: Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    logistic_model.fit(X_train, y_train)

    # Model 2: Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest_model.fit(X_train, y_train)

    # Save trained models and preprocessing tool
    joblib.dump(
        logistic_model,
        model_dir / "logistic_model.pkl"
    )

    joblib.dump(
        random_forest_model,
        model_dir / "random_forest_model.pkl"
    )

    joblib.dump(
        scaler,
        model_dir / "scaler.pkl"
    )

    print("Both models trained successfully.")
    print("Models and scaler saved in the models folder.")


if __name__ == "__main__":
    train_models()
