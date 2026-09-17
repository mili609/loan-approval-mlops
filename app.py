"""
FastAPI serving layer for the Loan Approval Prediction System.

Loads the trained preprocessing + model Pipeline (models/loan_approval_model.joblib)
once at startup and exposes it as a REST API. The same pipeline used during
training is reused here, so incoming requests get identical preprocessing.

Run locally:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.training.model_training import load_model_artifact

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan application will be approved.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the training pipeline first (e.g. 'dvc repro' or "
            "'python -m src.training.mlflow_tracking')."
        )
    model = load_model_artifact(MODEL_PATH)


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_df = pd.DataFrame([application.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return PredictionResponse(
        loan_status="Approved" if prediction == 1 else "Rejected",
        approved=bool(prediction == 1),
        approval_probability=round(float(probability), 4),
    )
