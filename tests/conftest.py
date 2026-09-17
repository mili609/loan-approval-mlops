"""Shared pytest fixtures for the Loan Approval Prediction System tests."""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"


@pytest.fixture
def valid_loan_dataframe() -> pd.DataFrame:
    """A small, schema-valid dataset: correct columns, no duplicate rows,
    no missing target values, but with some missing feature values so it
    also exercises the imputers in the preprocessing pipeline.
    """
    return pd.DataFrame({
        "loan_id": ["lp001001", "lp001002", "lp001003", "lp001004"],
        "gender": ["male", "female", "male", None],
        "married": ["yes", "no", "yes", "no"],
        "dependents": ["0", "1", "0", "2"],
        "education": ["graduate", "not graduate", "graduate", "graduate"],
        "self_employed": ["no", "no", "yes", None],
        "applicantincome": [5000, 3000, 4000, 2500],
        "coapplicantincome": [0.0, 1500.0, 0.0, 0.0],
        "loanamount": [120.0, None, 100.0, 80.0],
        "loan_amount_term": [360.0, 360.0, None, 180.0],
        "credit_history": [1.0, 0.0, 1.0, None],
        "property_area": ["urban", "rural", "semiurban", "urban"],
        "loan_status": ["y", "n", "y", "n"],
    })


@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    """The project's real raw dataset, loaded read-only. Tests must not
    write to RAW_DATA_PATH; this fixture only reads it.
    """
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Raw dataset not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture(scope="session")
def model_path() -> Path:
    return MODEL_PATH


@pytest.fixture(scope="session")
def trained_pipeline():
    """The trained preprocessing+model Pipeline, loaded once per test session."""
    if not MODEL_PATH.exists():
        pytest.skip(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the training pipeline first (e.g. 'dvc repro')."
        )
    from src.training.model_training import load_model_artifact

    return load_model_artifact(MODEL_PATH)


@pytest.fixture
def sample_applicant() -> pd.DataFrame:
    """A single realistic applicant row matching the API/feature schema
    (no loan_id / loan_status, since those are dropped before modeling).
    """
    return pd.DataFrame([{
        "gender": "male",
        "married": "yes",
        "dependents": "0",
        "education": "graduate",
        "self_employed": "no",
        "applicantincome": 5849,
        "coapplicantincome": 0.0,
        "loanamount": 128.0,
        "loan_amount_term": 360.0,
        "credit_history": 1.0,
        "property_area": "urban",
    }])
