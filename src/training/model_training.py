"""
Model training and evaluation for the Loan Approval Prediction System.

Trains multiple candidate models on the same preprocessed data so they
can be compared fairly, evaluates each with several metrics (accuracy
alone is not a reliable choice for an imbalanced target), and saves
the best-performing model as a single artifact that bundles
preprocessing + the trained classifier.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.transformation.data_transformation import build_preprocessing_pipeline

RANDOM_STATE = 42

# Candidate models to train and compare. Kept as a function so a fresh,
# unfitted estimator is created every time it is called.
def get_candidate_models() -> dict:
    """Return a dict of {model_name: unfitted estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=100
        ),
    }


def build_model_pipeline(estimator) -> Pipeline:
    """Combine preprocessing and an estimator into a single Pipeline.

    Bundling preprocessing with the model means the exact same
    transformations used at training time are automatically applied
    to new data at prediction time (e.g. from an API).
    """
    return Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", estimator),
    ])


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit every candidate model on the training data.

    Returns a dict of {model_name: fitted Pipeline}.
    """
    fitted_models = {}

    for name, estimator in get_candidate_models().items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    return fitted_models


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute accuracy, precision, recall, F1-score and confusion matrix."""
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def evaluate_models(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate every fitted model. Returns {model_name: metrics_dict}."""
    return {
        name: evaluate_model(pipeline, X_test, y_test)
        for name, pipeline in fitted_models.items()
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a clean side-by-side comparison table of all models' metrics.

    The confusion matrix is excluded from the table (it is not a
    scalar) and should be inspected separately per model.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    return pd.DataFrame(rows).set_index("model").round(4)


def select_best_model(results: dict, fitted_models: dict, metric: str = "f1_score"):
    """Pick the best model by a chosen metric (F1-score by default).

    F1-score balances precision and recall, which is more informative
    than accuracy alone on an imbalanced target like loan_status.
    Returns (best_model_name, best_pipeline).
    """
    best_name = max(results, key=lambda name: results[name][metric])
    return best_name, fitted_models[best_name]


def save_model_artifact(pipeline: Pipeline, output_path: str | Path) -> None:
    """Save the fitted preprocessing + model pipeline as a single joblib file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def load_model_artifact(model_path: str | Path) -> Pipeline:
    """Load a previously saved preprocessing + model pipeline."""
    return joblib.load(model_path)
