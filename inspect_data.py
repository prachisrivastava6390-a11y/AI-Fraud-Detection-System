import pandas as pd
from pathlib import Path

data_path = Path("Data/creditcard.csv")

print("Checking dataset location...")

if not data_path.exists():
    print("Dataset not found at:", data_path.resolve())
    print("Please check the Data folder and CSV filename.")

else:
    df = pd.read_csv(data_path)

    print("\n===== DATASET LOADED SUCCESSFULLY =====")
    print("Dataset shape:", df.shape)

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum().sum())

    print("\n===== FRAUD DISTRIBUTION =====")
    print(df["Class"].value_counts())

    print("\n===== CLASS DISTRIBUTION (%) =====")
    print(
        (df["Class"].value_counts(normalize=True) * 100).round(4)
    )
