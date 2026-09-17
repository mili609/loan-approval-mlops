"""
Data transformation and feature engineering for the Loan Approval
Prediction System.

This module builds a reusable, leakage-safe preprocessing pipeline:
- Splits raw data into features (X) and target (y).
- Drops identifier columns that should not be used for modeling.
- Encodes the target column (y/n -> 1/0).
- Builds a sklearn ColumnTransformer that imputes missing values and
  encodes categorical features.

The same functions/objects are reused by the training pipeline and,
later, by the prediction/API code, so preprocessing is guaranteed to
be identical everywhere.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that only identify a record and carry no predictive signal.
ID_COLUMNS = ["loan_id"]

TARGET_COLUMN = "loan_status"

# Mapping used to convert the target column to a binary numeric label.
TARGET_MAPPING = {"y": 1, "n": 0}

NUMERICAL_COLUMNS = [
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "loan_amount_term",
    "credit_history",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "property_area",
]


def encode_target(y: pd.Series) -> pd.Series:
    """Convert the loan_status column ('y'/'n') into binary labels (1/0)."""
    y_normalized = y.astype(str).str.strip().str.lower()
    y_encoded = y_normalized.map(TARGET_MAPPING)

    if y_encoded.isnull().any():
        unknown_values = y_normalized[y_encoded.isnull()].unique()
        raise ValueError(f"Unexpected loan_status values: {unknown_values}")

    return y_encoded.astype(int)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the raw dataframe into model features (X) and target (y).

    The identifier column (loan_id) is dropped since it has no
    predictive value, and the target is encoded to 1/0.
    """
    y = encode_target(df[TARGET_COLUMN])
    X = df.drop(columns=ID_COLUMNS + [TARGET_COLUMN])

    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and encodes raw features.

    Numerical columns: median imputation + standard scaling.
    Categorical columns: most-frequent imputation + one-hot encoding.

    Fitting is deferred to training data only, so no information from
    the test set leaks into the transformation.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])

    return preprocessor


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features/target into train and test sets.

    Splitting happens before any preprocessing is fit, and the split
    is stratified on the target to preserve the class balance since
    loan_status is imbalanced (~69% approved).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_raw_dataset(raw_data_path: str | Path) -> pd.DataFrame:
    """Load the raw loan dataset from disk without modifying it."""
    return pd.read_csv(raw_data_path)


def save_processed_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    processed_dir: str | Path,
) -> None:
    """Save the train/test split (pre-encoding, post target-mapping) to disk.

    Saving the split before ColumnTransformer encoding keeps the CSVs
    human-readable; the fitted encoder itself is saved separately as
    part of the model artifact in the training step.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.assign(loan_status=y_train.values).to_csv(
        processed_dir / "train.csv", index=False
    )
    X_test.assign(loan_status=y_test.values).to_csv(
        processed_dir / "test.csv", index=False
    )


def prepare_train_test_data(
    raw_data_path: str | Path,
    processed_dir: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run the full transformation step: load, split features/target,
    train/test split, and optionally persist the processed CSVs.

    Returns X_train, X_test, y_train, y_test.
    """
    df = load_raw_dataset(raw_data_path)
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    if processed_dir is not None:
        save_processed_datasets(X_train, X_test, y_train, y_test, processed_dir)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    prepare_train_test_data(
        raw_data_path=PROJECT_ROOT / "data" / "raw" / "loan_data.csv",
        processed_dir=PROJECT_ROOT / "data" / "processed",
    )
    print("Train/test split saved to data/processed/ ✅")
