# Loan Approval Prediction — MLOps Project

Predicts whether a loan application will be approved, built as an
end-to-end MLOps pipeline: data validation, versioned data (DVC),
experiment tracking (MLflow), and a containerized prediction API
(FastAPI).

## Project Structure

```
├── app.py                      # FastAPI serving app
├── dvc.yaml                    # DVC pipeline: validate -> prepare -> train
├── Dockerfile                  # Container image for the API
├── data/
│   ├── raw/loan_data.csv       # Raw dataset (DVC-tracked, not in git)
│   └── processed/              # Train/test split produced by the pipeline
├── models/loan_approval_model.joblib  # Best model (preprocessing + classifier)
├── mlruns/, mlflow.db          # Local MLflow tracking store
├── notebooks/loan_approval.ipynb      # Exploratory analysis + pipeline walkthrough
└── src/
    ├── validation/data_validation.py       # Schema/target/duplicate checks
    ├── transformation/data_transformation.py  # Preprocessing + train/test split
    └── training/
        ├── model_training.py       # Train/evaluate/select/save candidate models
        └── mlflow_tracking.py      # Logs each candidate model as an MLflow run
```

## Dataset

614 loan applications with 13 columns (`data/raw/loan_data.csv`), target
`loan_status` (`y`/`n`, ~69% approved / 31% rejected). Several columns
have missing values, which the preprocessing pipeline imputes
(median for numerical, most-frequent for categorical).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Pull the raw data (tracked with DVC):

```bash
dvc pull
```

## Running the Pipeline

The pipeline has three stages, defined in `dvc.yaml`:

1. **validate** — checks required columns, target validity, and duplicates.
2. **prepare** — splits features/target, does a stratified train/test split,
   and writes `data/processed/train.csv` / `test.csv`.
3. **train** — trains Logistic Regression and Random Forest, evaluates both
   on the held-out test set, logs each as an MLflow run, and saves the
   best model (by F1-score) to `models/loan_approval_model.joblib`.

Run the full pipeline:

```bash
dvc repro
```

Or run stages individually:

```bash
python -m src.validation.data_validation
python -m src.transformation.data_transformation
python -m src.training.mlflow_tracking
```

Inspect experiment results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open http://127.0.0.1:5000.

### Current results

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.8618   | 0.8400    | 0.9882 | **0.9081** |
| Random Forest        | 0.8211   | 0.8462    | 0.9059 | 0.8750   |

Logistic Regression is selected as the best model (highest F1-score, which
matters more than accuracy given the class imbalance).

## Serving Predictions

Start the API locally:

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for interactive Swagger docs, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "male", "married": "yes", "dependents": "0",
        "education": "graduate", "self_employed": "no",
        "applicantincome": 5849, "coapplicantincome": 0,
        "loanamount": 128, "loan_amount_term": 360,
        "credit_history": 1, "property_area": "urban"
      }'
```

Response:

```json
{"loan_status": "Approved", "approved": true, "approval_probability": 0.815}
```

`GET /health` reports whether the model artifact loaded successfully.

## Docker

```bash
docker build -t loan-approval-api .
docker run -p 8000:8000 loan-approval-api
```

The image bundles the trained model artifact, so `models/loan_approval_model.joblib`
must exist (run the training pipeline) before building.

## Tech Stack

pandas, scikit-learn, joblib — data processing and modeling
DVC — data versioning and pipeline reproducibility
MLflow — experiment tracking
FastAPI + uvicorn — model serving
Docker — containerization
