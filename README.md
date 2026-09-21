# Loan Approval Prediction — End-to-End MLOps System

## 1. Project Overview

This project implements an **end-to-end MLOps pipeline** for predicting loan approval outcomes. It goes beyond a standalone ML model and covers the full lifecycle of a production ML system:

- Data versioning with **DVC**
- Automated data validation
- Feature engineering and model training with **scikit-learn**
- Experiment tracking with **MLflow**
- Automated testing with **Pytest**
- Containerization with **Docker**
- Continuous Integration with **GitHub Actions**
- Deployment to **Kubernetes**
- A **FastAPI** serving layer
- Monitoring with **Prometheus** and **Grafana**

The goal is to demonstrate how a machine learning model moves from raw data to a monitored, containerized, and orchestrated production service.

## 2. Project Structure

```
loan-approval-mlops/
├── data/
│   ├── raw/                     # Raw dataset, tracked with DVC
│   │   ├── loan_data.csv
│   │   └── loan_data.csv.dvc
│   └── processed/               # Train/test splits produced by the pipeline
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   └── loan_approval.ipynb      # Exploratory data analysis
├── src/
│   ├── validation/
│   │   └── data_validation.py   # Schema, target and duplicate checks
│   ├── transformation/
│   │   └── data_transformation.py  # Feature engineering pipeline
│   └── training/
│       ├── model_training.py       # Train/evaluate/select/save models
│       └── mlflow_tracking.py      # MLflow experiment logging
├── tests/
│   ├── test_data_validation.py
│   ├── test_data_transformation.py
│   ├── test_model.py
│   └── test_api.py
├── deployment/
│   └── kubernetes/
│       ├── deployment.yaml      # FastAPI Kubernetes Deployment
│       └── service.yaml         # FastAPI Kubernetes Service (NodePort)
├── monitoring/
│   ├── prometheus.yml               # Prometheus scrape config
│   ├── prometheus-configmap.yaml    # Prometheus config as a K8s ConfigMap
│   ├── prometheus-deployment.yaml   # Prometheus Kubernetes Deployment
│   ├── prometheus-service.yaml      # Prometheus Kubernetes Service
│   └── grafana/
│       ├── deployment.yaml          # Grafana Kubernetes Deployment
│       └── service.yaml             # Grafana Kubernetes Service
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
├── models/
│   └── loan_approval_model.joblib   # Best model artifact (pipeline + model)
├── app.py                        # FastAPI application
├── Dockerfile                    # Container image for the FastAPI service
├── dvc.yaml                      # DVC pipeline stages (validate → prepare → train)
├── requirements.txt
└── README.md
```

## 3. MLOps Pipeline

The system follows this pipeline flow:

```
Data → DVC → Validation → Feature Engineering → Model Training → MLflow
     → Pytest → Docker → GitHub Actions → Kubernetes → FastAPI
     → Prometheus → Grafana
```

1. Raw loan data is versioned with **DVC**.
2. **Validation** checks schema, target integrity, and duplicates before anything downstream runs.
3. **Feature engineering** (imputation, scaling, one-hot encoding) is built as a reusable scikit-learn pipeline.
4. **Model training** fits and compares candidate models.
5. **MLflow** logs parameters, metrics, and artifacts for each candidate run.
6. **Pytest** verifies validation, transformation, training, and API behavior.
7. **Docker** packages the FastAPI serving app and trained model into a single image.
8. **GitHub Actions** runs tests and builds the Docker image on every push/PR.
9. **Kubernetes** deploys the containerized API.
10. **FastAPI** serves real-time predictions and exposes a `/metrics` endpoint.
11. **Prometheus** scrapes those metrics, and **Grafana** visualizes them on a dashboard.

## 4. Data Versioning (DVC)

The raw loan dataset (`data/raw/loan_data.csv`) is tracked using **DVC** (`data/raw/loan_data.csv.dvc`), keeping large data files out of Git while still versioning them alongside the code. The DVC pipeline is defined in [dvc.yaml](dvc.yaml) with three stages: `validate`, `prepare` (feature engineering/train-test split), and `train`.

## 5. Data Validation

Before training, [src/validation/data_validation.py](src/validation/data_validation.py) runs automated checks on the raw dataset:

- **Schema validation** — confirms all required columns are present.
- **Target validation** — ensures the `loan_status` target column exists and has no missing values.
- **Duplicate check** — verifies there are no duplicate records.

The pipeline fails fast with a clear error if any check does not pass.

## 6. Model Training

Two candidate models are trained on the same preprocessed data (imputation, scaling, one-hot encoding) and evaluated on a held-out test set:

**Logistic Regression**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8618 |
| Precision | 0.8400 |
| Recall    | 0.9882 |
| F1-score  | 0.9081 |

**Random Forest**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8211 |
| Precision | 0.8462 |
| Recall    | 0.9059 |
| F1-score  | 0.8750 |

**Logistic Regression was selected as the final model based on its higher F1-score**, since F1 balances precision and recall — a more reliable choice than raw accuracy on this imbalanced target. The selected model is saved as [models/loan_approval_model.joblib](models/loan_approval_model.joblib).

## 7. MLflow Experiment Tracking

Both candidate models (Logistic Regression and Random Forest) are tracked as separate **MLflow runs** within a single experiment (`Loan Approval Prediction`). Each run logs:

- Model hyperparameters
- Evaluation metrics (accuracy, precision, recall, F1-score)
- A confusion matrix artifact
- The fitted preprocessing + model pipeline

Tracking uses a local SQLite store (`mlflow.db`) with artifacts in `mlruns/`, so no external MLflow server is required. View runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 8. Testing

The project uses **Pytest** for automated testing:

- **35 tests passed**
- **76% code coverage**
- Tests cover data validation, data transformation, model training, and FastAPI API behavior (`tests/test_data_validation.py`, `tests/test_data_transformation.py`, `tests/test_model.py`, `tests/test_api.py`).

Run the test suite with coverage:

```bash
pytest -v --cov=src --cov=app
```

## 9. FastAPI Service

The trained model is served via a FastAPI application ([app.py](app.py)):

| Method | Endpoint    | Description                                      |
|--------|-------------|---------------------------------------------------|
| GET    | `/`         | API info (name, docs link, metrics link)          |
| GET    | `/health`   | Health check + model load status                  |
| POST   | `/predict`  | Predicts loan approval for a given application    |
| GET    | `/metrics`  | Prometheus-formatted metrics                       |

Interactive API documentation (Swagger UI) is available at **`/docs`**.

Run locally:

```bash
uvicorn app:app --reload
```

## 10. Docker

Build and run the FastAPI service in a container:

```bash
docker build -t loan-approval-api .
docker run -p 8000:8000 loan-approval-api
```

## 11. GitHub Actions (CI)

Continuous Integration is defined at [.github/workflows/ci.yml](.github/workflows/ci.yml) and runs automatically on every push/PR to `master`:

1. Install dependencies from `requirements.txt`.
2. Run the **Pytest** suite with coverage (`pytest -v --cov=src --cov=app`).
3. Build the **Docker** image (`docker build -t loan-approval-api .`).

This ensures every change is tested and remains containerizable before merging.

## 12. Kubernetes Deployment

The FastAPI application is deployed to Kubernetes using the manifests in [deployment/kubernetes/](deployment/kubernetes/):

- **[deployment.yaml](deployment/kubernetes/deployment.yaml)** — a Kubernetes **Deployment** (`loan-approval-api`) running **1 replica** of the container on port `8000`.
- **[service.yaml](deployment/kubernetes/service.yaml)** — a Kubernetes **Service** (`loan-approval-service`) exposing port `8000` via **NodePort**.

Deploy with:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

The `/health` endpoint was tested successfully against the running Kubernetes service, confirming the deployed pod loads the model and serves traffic correctly.

## 13. Monitoring (Prometheus + Grafana)

The FastAPI app exposes a **`/metrics`** endpoint (via `prometheus-client`) with custom metrics:

- `loan_prediction_requests_total` — total prediction requests
- `loan_approved_total` — total approved predictions
- `loan_rejected_total` — total rejected predictions
- `loan_prediction_latency_seconds` — prediction latency histogram

**Prometheus** is deployed on Kubernetes ([monitoring/prometheus-deployment.yaml](monitoring/prometheus-deployment.yaml), [monitoring/prometheus-service.yaml](monitoring/prometheus-service.yaml)) and configured via [monitoring/prometheus-configmap.yaml](monitoring/prometheus-configmap.yaml) / [monitoring/prometheus.yml](monitoring/prometheus.yml) to scrape `loan-approval-service:8000/metrics` every 15 seconds.

**Grafana** is deployed on Kubernetes ([monitoring/grafana/deployment.yaml](monitoring/grafana/deployment.yaml), [monitoring/grafana/service.yaml](monitoring/grafana/service.yaml)) with **Prometheus configured as its datasource**. The Grafana dashboard visualizes:

- Total Loan Predictions
- Approved Loans
- Rejected Loans
- Prediction Latency

## 14. Reproducibility

Clone the repository and reproduce the full pipeline end-to-end:

```bash
# Run the DVC pipeline (validate → prepare → train)
dvc repro

# Run the test suite
pytest -v --cov=src --cov=app

# Build the Docker image
docker build -t loan-approval-api .

# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

## 15. Tech Stack

- **Python**
- **pandas**
- **scikit-learn**
- **DVC**
- **MLflow**
- **Pytest**
- **FastAPI**
- **Docker**
- **GitHub Actions**
- **Kubernetes**
- **Prometheus**
- **Grafana**
