import joblib
import pandas as pd
import numpy as np

def predict_point(input_data):
    # Load model and features
    model = joblib.load('src/model.joblib')
    features_list = joblib.load('src/features_list.joblib')
    
    # Map input data to model features
    # Note: Many features are missing from input, using defaults
    data_dict = {
        'volt': input_data.get('voltage', 0.0),
        'rotate': input_data.get('rotation', 0.0),
        'pressure': input_data.get('pressure', 0.0),
        'vibration': input_data.get('vibration', 0.0),
        'volt_mean_24h': input_data.get('voltage', 0.0), # Default to current
        'volt_std_24h': 0.0,
        'rotate_mean_24h': input_data.get('rotation', 0.0),
        'rotate_std_24h': 0.0,
        'pressure_mean_24h': input_data.get('pressure', 0.0),
        'pressure_std_24h': 0.0,
        'vibration_mean_24h': input_data.get('vibration_24h_mean', 37.8),
        'vibration_std_24h': input_data.get('vibration_24h_std', 2.9),
        'volt_lag_3h': input_data.get('voltage', 0.0),
        'volt_change_3h': 0.0,
        'rotate_lag_3h': input_data.get('rotation', 0.0),
        'rotate_change_3h': 0.0,
        'pressure_lag_3h': input_data.get('pressure', 0.0) - input_data.get('pressure_change_3h', 4.6),
        'pressure_change_3h': input_data.get('pressure_change_3h', 4.6),
        'vibration_lag_3h': input_data.get('vibration', 0.0),
        'vibration_change_3h': 0.0,
        'days_since_comp1': input_data.get('days_since_maintenance', 15),
        'days_since_comp2': input_data.get('days_since_maintenance', 15),
        'days_since_comp3': input_data.get('days_since_maintenance', 15),
        'days_since_comp4': input_data.get('days_since_maintenance', 15)
    }
    
    df = pd.DataFrame([data_dict])
    X = df[features_list]
    
    prob = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]
    
    return prob, pred

if __name__ == "__main__":
    user_input = {
      "machineID": 12,
      "voltage": 172.4,
      "rotation": 456.7,
      "pressure": 118.2,
      "vibration": 39.5,
      "vibration_24h_mean": 37.8,
      "vibration_24h_std": 2.9,
      "pressure_change_3h": 4.6,
      "days_since_maintenance": 15
    }
    
    prob, pred = predict_point(user_input)
    print(f"Prob: {prob:.4f}, Pred: {pred}")
