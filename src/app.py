from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Predictive Maintenance API")

# Load model and feature list
MODEL_PATH = 'src/model.joblib'
FEATURES_PATH = 'src/features_list.joblib'

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model = joblib.load(MODEL_PATH)
    features_list = joblib.load(FEATURES_PATH)
else:
    model = None
    features_list = None
    print("Warning: Model or features list not found. Transitions to /predict will fail.")

class SensorReadings(BaseModel):
    # This model expects the engineered features
    # In a real scenario, we'd take raw data and compute these,
    # but for the API endpoint, we provide a structured way to pass them.
    machine_id: int
    volt: float
    rotate: float
    pressure: float
    vibration: float
    # Optional: allow passing precomputed features
    volt_mean_24h: float = 0.0
    volt_std_24h: float = 0.0
    rotate_mean_24h: float = 0.0
    rotate_std_24h: float = 0.0
    pressure_mean_24h: float = 0.0
    pressure_std_24h: float = 0.0
    vibration_mean_24h: float = 0.0
    vibration_std_24h: float = 0.0
    volt_lag_3h: float = 0.0
    volt_change_3h: float = 0.0
    rotate_lag_3h: float = 0.0
    rotate_change_3h: float = 0.0
    pressure_lag_3h: float = 0.0
    pressure_change_3h: float = 0.0
    vibration_lag_3h: float = 0.0
    vibration_change_3h: float = 0.0
    days_since_comp1: float = 0.0
    days_since_comp2: float = 0.0
    days_since_comp3: float = 0.0
    days_since_comp4: float = 0.0

@app.get("/")
def read_root():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict")
def predict(data: SensorReadings):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Create a DataFrame from the input
    input_dict = data.dict()
    # Remove machine_id as it wasn't a feature (it was dropped in train.py)
    # Actually, check what features were used
    df_input = pd.DataFrame([input_dict])
    
    # Ensure all required features are present and in the right order
    try:
        X = df_input[features_list]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {str(e)}")
    
    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    
    return {
        "machine_id": data.machine_id,
        "failure_probability": prob,
        "predicted_class": pred,
        "status": "Potential Failure Detected" if pred == 1 else "Normal Operations"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
