"""
MLflow experiment tracking for the Loan Approval Prediction System.

Logs each candidate model (Logistic Regression, Random Forest) as its
own MLflow run inside a single experiment: hyperparameters, evaluation
metrics, a confusion matrix artifact, and the fitted preprocessing +
model Pipeline itself. Uses a local file-based tracking store
(./mlruns at the project root), so no remote server or cloud
credentials are required.

Run directly to train + log both models:
    python -m src.training.mlflow_tracking
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed for saving PNG artifacts
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.training.model_training import (
    evaluate_models,
    save_model_artifact,
    select_best_model,
    train_models,
)
from src.transformation.data_transformation import prepare_train_test_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"
EXPERIMENT_NAME = "Loan Approval Prediction"

# Hyperparameters worth recording per model type (kept short and relevant
# rather than dumping every internal sklearn default).
HYPERPARAMETER_KEYS = {
    LogisticRegression: ["C", "max_iter", "penalty", "solver", "random_state"],
    RandomForestClassifier: [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    ],
}


def configure_mlflow() -> None:
    """Point MLflow at a local SQLite tracking store and select the experiment.

    A local sqlite file (mlflow.db) plus a local artifact folder
    (mlruns/) means tracking works fully offline, with no MLflow
    server or cloud account required. SQLite is used instead of the
    plain file store because newer MLflow versions put the legacy file
    store in maintenance mode.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=f"file:{MLRUNS_DIR}")

    mlflow.set_experiment(EXPERIMENT_NAME)


def get_hyperparameters(estimator) -> dict:
    """Extract a small, relevant set of hyperparameters for a given estimator."""
    keys = HYPERPARAMETER_KEYS.get(type(estimator), [])
    params = estimator.get_params()
    return {key: params[key] for key in keys if key in params}


def save_confusion_matrix_plot(confusion_matrix, model_name: str, output_path: Path) -> Path:
    """Render a confusion matrix as a PNG image for logging as an MLflow artifact."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
    ax.set_yticklabels(["Rejected (0)", "Approved (1)"])

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i, str(confusion_matrix[i, j]),
                ha="center", va="center", color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def log_model_run(
    model_name: str,
    pipeline: Pipeline,
    metrics: dict,
    is_best: bool,
    X_train: pd.DataFrame,
) -> str:
    """Log a single trained model as one MLflow run. Returns the run ID."""
    estimator = pipeline.named_steps["model"]

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(get_hyperparameters(estimator))

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        mlflow.set_tag("is_best_model", "yes" if is_best else "no")

        with TemporaryDirectory() as tmp_dir:
            plot_path = save_confusion_matrix_plot(
                metrics["confusion_matrix"], model_name, Path(tmp_dir) / "confusion_matrix.png"
            )
            mlflow.log_artifact(str(plot_path))

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_train.head(2),
            # "pickle" keeps this consistent with the joblib artifact saved
            # elsewhere in the project; MLflow's newer default (skops) is
            # stricter about which types it trusts on load.
            serialization_format="pickle",
        )

        return run.info.run_id


def run_mlflow_tracking(raw_data_path: str | Path = None) -> dict:
    """Train both candidate models and log each as an MLflow run.

    Reuses the existing transformation/training functions unchanged, so
    model-selection logic (highest F1-score) is not duplicated here.
    Returns a dict describing the experiment, runs, and the best model.
    """
    if raw_data_path is None:
        raw_data_path = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"

    X_train, X_test, y_train, y_test = prepare_train_test_data(raw_data_path)

    fitted_models = train_models(X_train, y_train)
    results = evaluate_models(fitted_models, X_test, y_test)
    best_name, _ = select_best_model(results, fitted_models, metric="f1_score")

    configure_mlflow()

    run_ids = {}
    for model_name, pipeline in fitted_models.items():
        run_ids[model_name] = log_model_run(
            model_name=model_name,
            pipeline=pipeline,
            metrics=results[model_name],
            is_best=(model_name == best_name),
            X_train=X_train,
        )

    # Also save the best model as a standalone joblib artifact (independent
    # of the MLflow store) so it can be loaded directly by the serving API.
    save_model_artifact(fitted_models[best_name], MODEL_OUTPUT_PATH)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_ids": run_ids,
        "best_model": best_name,
        "results": results,
        "model_path": str(MODEL_OUTPUT_PATH),
    }


if __name__ == "__main__":
    summary = run_mlflow_tracking()

    print(f"Experiment: {summary['experiment_name']}")
    print(f"Tracking URI: {summary['tracking_uri']}")
    print("\nRuns logged:")
    for model_name, run_id in summary["run_ids"].items():
        marker = " (best)" if model_name == summary["best_model"] else ""
        print(f"  - {model_name}{marker}: run_id={run_id}")

    print(f"\nBest model: {summary['best_model']}")
    print(f"Model artifact saved to: {summary['model_path']}")
    print(
        "\nRun 'mlflow ui --backend-store-uri sqlite:///mlflow.db' from the "
        "project root and open http://127.0.0.1:5000 to inspect runs."
    )
