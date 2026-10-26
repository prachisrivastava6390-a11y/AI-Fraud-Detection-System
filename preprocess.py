import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path):
    """Load, prepare, and split transaction data."""

    data_path = Path(file_path)

    if not data_path.exists():
        print(f"Dataset not found: {data_path.resolve()}")
        return None

    # Load the dataset
    df = pd.read_csv(data_path)

    # Check whether the target column exists
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    # Separate input features and target label
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Split data before fitting the scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numerical features using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":
    DATA_PATH = Path("Data") / "creditcard.csv"
    preprocess_data(DATA_PATH)
