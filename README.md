# Predictive Maintenance Pipeline

This project implements an end-to-end pipeline to predict machine failures based on telemetry, maintenance logs, and failure records.

## Project Structure
- `src/preprocess.py`: Processes raw data and engineers rolling statistics, recency, and lag features.
- `src/train.py`: Trains a baseline Random Forest model and an optimized XGBoost model.
- `src/app.py`: Serves the XGBoost model via a FastAPI REST endpoint.
- `tests/test_api.py`: Verification script for the API.
- `report.md`: Detailed explanation of the evaluation metrics used.

## Getting Started

### 1. Prerequisites
Install the required libraries:
```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib httpx
```

### 2. Prepare Data
Run the preprocessing script to clean and join the datasets:
```bash
python src/preprocess.py
```
This generates `d:\Jitendra\Data\processed_data.csv`.

### 3. Train Model
Train the models and save the best one:
```bash
python src/train.py
```
This saves `src/model.joblib` and generates `report.md`.

### 4. Serve API
Start the FastAPI server:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 6. Evaluate New Data
To test Recall and ROC-AUC for new data points:
1. Ensure your new raw CSV files (telemetry, etc.) include the actual outcomes (failures).
2. Run `src/preprocess.py` on the new data to generate engineered features and label the `target` column.
3. Run the evaluation script:
```bash
python src/evaluate.py
```
This script loads the trained model and compares its predictions against the ground truth labels in your dataset, outputting Recall, ROC-AUC, and a confusion matrix.

### 7. Run Streamlit UI (Frontend)
To test the API interactively:
1. Ensure the FastAPI server is running (`uvicorn src.app:app`).
2. Run the Streamlit dashboard:
```bash
streamlit run src/ui.py
```
This will open a browser window where you can adjust sensor values and see real-time failure probabilities.
**Payload Example**:
```json
{
  "machine_id": 1,
  "volt": 170.0,
  "rotate": 450.0,
  "pressure": 100.0,
  "vibration": 35.0,
  "volt_mean_24h": 170.0,
  "volt_std_24h": 10.0,
  "volt_lag_3h": 170.0,
  "volt_change_3h": 0.0,
  ... (see src/app.py for all fields)
}
```
*Note: The API provides default values for engineered features if omitted, but for best accuracy, use calculated features.*
