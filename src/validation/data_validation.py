import pandas as pd


REQUIRED_COLUMNS = [
    "loan_id",
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "loan_amount_term",
    "credit_history",
    "property_area",
    "loan_status",
]


def validate_columns(df: pd.DataFrame) -> bool:
    """Check whether all required columns are present."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def validate_target(df: pd.DataFrame) -> bool:
    """Check that the target column exists and has valid values."""
    if "loan_status" not in df.columns:
        raise ValueError("Target column 'loan_status' is missing.")

    if df["loan_status"].isnull().any():
        raise ValueError("Target column contains missing values.")

    return True


def validate_duplicates(df: pd.DataFrame) -> bool:
    """Check for duplicate records."""
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    return True


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run all dataset validation checks."""
    validate_columns(df)
    validate_target(df)
    validate_duplicates(df)

    print("Data validation passed successfully! ✅")
    return True


if __name__ == "__main__":
    from pathlib import Path

    raw_data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "loan_data.csv"
    validate_dataset(pd.read_csv(raw_data_path))