from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model and Label Encoder
with open("xgboost_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define input schema for API
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Crop Recommendation API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(input_data: CropInput):
    # Prepare input features
    features = np.array([[input_data.N, input_data.P, input_data.K, input_data.temperature,
                          input_data.humidity, input_data.ph, input_data.rainfall]])

    # Predict the best crop
    y_pred_encoded = xgb_model.predict(features)
    predicted_crop = label_encoder.inverse_transform(y_pred_encoded)[0]

    # Get probabilities for the best and second-best crops
    probabilities = xgb_model.predict_proba(features)[0]
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order

    best_crop = label_encoder.inverse_transform([sorted_indices[0]])[0]
    best_confidence = probabilities[sorted_indices[0]] * 100

    second_best_crop = label_encoder.inverse_transform([sorted_indices[1]])[0]
    second_best_confidence = probabilities[sorted_indices[1]] * 100

    return {
        "predicted_crop": predicted_crop,
        "confidence": {
            "best_crop": {
                "name": best_crop,
                "confidence": f"{best_confidence:.2f}%"
            },
            "second_best_crop": {
                "name": second_best_crop,
                "confidence": f"{second_best_confidence:.2f}%"
            }
        }
    }
