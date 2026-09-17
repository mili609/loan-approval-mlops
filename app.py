"""
FastAPI serving layer for the Loan Approval Prediction System.

Loads the trained preprocessing + model Pipeline once at startup and
exposes REST API endpoints for health checks, predictions, and Prometheus
monitoring metrics.

Run locally:
    uvicorn app:app --reload

API docs:
    http://127.0.0.1:8000/docs

Prometheus metrics:
    http://127.0.0.1:8000/metrics
"""

from pathlib import Path
from typing import Optional
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.training.model_training import load_model_artifact


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan application will be approved.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

prediction_requests = Counter(
    "loan_prediction_requests_total",
    "Total number of loan prediction requests",
)

prediction_latency = Histogram(
    "loan_prediction_latency_seconds",
    "Time spent processing loan predictions",
)

approved_predictions = Counter(
    "loan_approved_total",
    "Total number of approved loan predictions",
)

rejected_predictions = Counter(
    "loan_rejected_total",
    "Total number of rejected loan predictions",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

model = None


@app.on_event("startup")
def load_model() -> None:
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the training pipeline first "
            "(e.g. 'dvc repro' or "
            "'python -m src.training.mlflow_tracking')."
        )

    model = load_model_artifact(MODEL_PATH)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    gender: Optional[str] = Field(None, examples=["male"])
    married: Optional[str] = Field(None, examples=["yes"])
    dependents: Optional[str] = Field(None, examples=["0"])
    education: str = Field(..., examples=["graduate"])
    self_employed: Optional[str] = Field(None, examples=["no"])
    applicantincome: float = Field(..., examples=[5849])
    coapplicantincome: float = Field(0.0, examples=[0.0])
    loanamount: Optional[float] = Field(None, examples=[128.0])
    loan_amount_term: Optional[float] = Field(360.0, examples=[360.0])
    credit_history: Optional[float] = Field(None, examples=[1.0])
    property_area: str = Field(..., examples=["urban"])


class PredictionResponse(BaseModel):
    loan_status: str
    approved: bool
    approval_probability: float


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "message": "Loan Approval Prediction API",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    start_time = time.perf_counter()

    prediction_requests.inc()

    input_df = pd.DataFrame([application.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    prediction_latency.observe(
        time.perf_counter() - start_time
    )

    if prediction == 1:
        approved_predictions.inc()
    else:
        rejected_predictions.inc()

    return PredictionResponse(
        loan_status="Approved" if prediction == 1 else "Rejected",
        approved=bool(prediction == 1),
        approval_probability=round(float(probability), 4),
    )