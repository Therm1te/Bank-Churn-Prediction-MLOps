"""
model.py - Model Loading and Prediction Module
================================================
Loads the trained model.pkl and provides a prediction function.
"""

import os
import pickle
import numpy as np
import pandas as pd


# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")


def load_model():
    """Load the trained model artifact from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run the training script first: python notebooks/train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact


# Load model at module level for reuse
model_artifact = load_model()
model = model_artifact["model"]
feature_names = model_artifact["feature_names"]
model_name = model_artifact["model_name"]


def predict_churn(features_dict: dict) -> dict:
    """
    Predict whether a customer will churn based on input features.

    Parameters
    ----------
    features_dict : dict
        Dictionary with raw feature values:
        - CreditScore (int): 350-850
        - Geography (str): "France", "Germany", or "Spain"
        - Gender (str): "Male" or "Female"
        - Age (int): 18-92
        - Tenure (int): 0-10
        - Balance (float): >= 0
        - NumOfProducts (int): 1-4
        - HasCrCard (int): 0 or 1
        - IsActiveMember (int): 0 or 1
        - EstimatedSalary (float): >= 0

    Returns
    -------
    dict
        - prediction (int): 0 (Stayed) or 1 (Churned)
        - probability (float): Probability of churn
        - label (str): "Yes" or "No"
    """
    # Build a DataFrame with OneHotEncoded features to match training data
    input_data = {
        "CreditScore": [features_dict["CreditScore"]],
        "Age": [features_dict["Age"]],
        "Tenure": [features_dict["Tenure"]],
        "Balance": [features_dict["Balance"]],
        "NumOfProducts": [features_dict["NumOfProducts"]],
        "HasCrCard": [features_dict["HasCrCard"]],
        "IsActiveMember": [features_dict["IsActiveMember"]],
        "EstimatedSalary": [features_dict["EstimatedSalary"]],
    }

    # One-Hot encode Geography (drop_first=True → France is baseline)
    geography = features_dict.get("Geography", "France")
    input_data["Geography_Germany"] = [1 if geography == "Germany" else 0]
    input_data["Geography_Spain"] = [1 if geography == "Spain" else 0]

    # One-Hot encode Gender (drop_first=True → Female is baseline)
    gender = features_dict.get("Gender", "Male")
    input_data["Gender_Male"] = [1 if gender == "Male" else 0]

    # Create DataFrame with correct feature ordering
    df_input = pd.DataFrame(input_data)

    # Ensure columns match training feature order
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_names]

    # Make prediction
    prediction = int(model.predict(df_input)[0])
    probability = float(model.predict_proba(df_input)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Yes" if prediction == 1 else "No",
    }
