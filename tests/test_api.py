from fastapi.testclient import TestClient
from src.app import app
import json

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Predictive Maintenance API is running"}

def test_predict_normal():
    payload = {
        "machine_id": 1,
        "volt": 170.0,
        "rotate": 450.0,
        "pressure": 100.0,
        "vibration": 35.0,
        "volt_mean_24h": 170.0,
        "volt_std_24h": 10.0,
        "rotate_mean_24h": 450.0,
        "rotate_std_24h": 50.0,
        "pressure_mean_24h": 100.0,
        "pressure_std_24h": 10.0,
        "vibration_mean_24h": 35.0,
        "vibration_std_24h": 5.0,
        "volt_change_3h": 0.0,
        "rotate_change_3h": 0.0,
        "pressure_change_3h": 0.0,
        "vibration_change_3h": 0.0,
        "days_since_comp1": 10.0,
        "days_since_comp2": 10.0,
        "days_since_comp3": 10.0,
        "days_since_comp4": 10.0
    }
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}, Body: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    print(f"Normal prediction response: {data}")

def test_predict_failure_simulation():
    payload = {
        "machine_id": 1,
        "volt": 250.0,
        "rotate": 600.0,
        "pressure": 150.0,
        "vibration": 100.0,
        "volt_mean_24h": 220.0,
        "volt_std_24h": 30.0,
        "rotate_mean_24h": 550.0,
        "rotate_std_24h": 80.0,
        "pressure_mean_24h": 130.0,
        "pressure_std_24h": 20.0,
        "vibration_mean_24h": 80.0,
        "vibration_std_24h": 15.0,
        "volt_change_3h": 50.0,
        "rotate_change_3h": 100.0,
        "pressure_change_3h": 30.0,
        "vibration_change_3h": 20.0,
        "days_since_comp1": 100.0,
        "days_since_comp2": 100.0,
        "days_since_comp3": 100.0,
        "days_since_comp4": 100.0
    }
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}, Body: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    print(f"Failure simulation response: {data}")

if __name__ == "__main__":
    print("Running API tests...")
    test_read_root()
    test_predict_normal()
    test_predict_failure_simulation()
    print("All tests passed!")
