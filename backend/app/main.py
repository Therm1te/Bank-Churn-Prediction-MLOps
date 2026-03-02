"""
main.py - FastAPI Backend for Bank Churn Prediction
====================================================
Endpoints:
  GET  /        → Welcome message
  POST /predict → Predict customer churn from JSON input
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model import predict_churn, model_name

# ---- App Initialization ----
app = FastAPI(
    title="Bank Churn Prediction API",
    description=(
        "API to predict whether a bank customer will churn (leave) "
        "based on their profile features. Powered by a trained ML model."
    ),
    version="1.0.0",
)

# Enable CORS so Streamlit can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request / Response Schemas ----
class CustomerFeatures(BaseModel):
    """Input schema for customer features."""
    CreditScore: int = Field(..., ge=350, le=850, description="Customer's credit score (350-850)")
    Geography: str = Field(..., description="Country: France, Germany, or Spain")
    Gender: str = Field(..., description="Gender: Male or Female")
    Age: int = Field(..., ge=18, le=92, description="Customer's age (18-92)")
    Tenure: int = Field(..., ge=0, le=10, description="Years with the bank (0-10)")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0=No, 1=Yes)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0=No, 1=Yes)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated annual salary")

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 80000.0,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for churn prediction."""
    prediction: int = Field(..., description="0 = Stayed, 1 = Churned")
    probability: float = Field(..., description="Probability of churn (0-1)")
    label: str = Field(..., description="Human-readable: Yes or No")
    model_used: str = Field(..., description="Name of the ML model used")


# ---- Routes ----
@app.get("/", tags=["Health"])
def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Bank Churn Prediction API! 🏦",
        "model": model_name,
        "docs": "Visit /docs for interactive API documentation.",
        "predict_endpoint": "POST /predict",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Predict whether a customer will churn.

    Send customer features as JSON and receive a churn prediction.
    """
    # Validate categorical inputs
    if features.Geography not in ["France", "Germany", "Spain"]:
        raise HTTPException(
            status_code=400,
            detail="Geography must be one of: France, Germany, Spain",
        )
    if features.Gender not in ["Male", "Female"]:
        raise HTTPException(
            status_code=400,
            detail="Gender must be one of: Male, Female",
        )

    try:
        result = predict_churn(features.model_dump())
        result["model_used"] = model_name
        print(f"[PREDICT] Request: {features.model_dump()} -> {result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
