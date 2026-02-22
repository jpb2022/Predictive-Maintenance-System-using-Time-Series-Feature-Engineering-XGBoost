import streamlit as st
import requests
import json
import pandas as pd
import joblib
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🛠️ Machine Failure Prediction Dashboard")

# Load model and features for batch processing
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('src/model.joblib')
    features = joblib.load('src/features_list.joblib')
    return model, features

model, features_list = load_model_artifacts()

def make_prediction_api(url, payload):
    try:
        with st.spinner("Calling API..."):
            response = requests.post(url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            prob = result['failure_probability']
            status = result['status']
            
            st.divider()
            st.subheader("Prediction Result")
            if prob > 0.5:
                st.error(f"⚠️ {status}")
            else:
                st.success(f"✅ {status}")
            
            m1, m2 = st.columns(2)
            m1.metric("Failure Probability", f"{prob*100:.2f}%")
            m2.metric("Predicted Class", result.get('predicted_class', 'N/A'))
            
            st.write("Confidence Level")
            st.progress(prob)
            
        else:
            st.error(f"Error: Received status code {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.info("Make sure the FastAPI server is running: `uvicorn src.app:app --reload`")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Single Prediction", "Batch Evaluation"])

with tab1:
    st.sidebar.header("API Configuration")
    api_url = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000/predict")

    input_method = st.radio("Select Input Method", ["Manual sliders", "Raw JSON Payload"])

    if input_method == "Manual sliders":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sensor Input")
            machine_id = st.number_input("Machine ID", min_value=1, value=1)
            with st.expander("Real-time Sensors", expanded=True):
                volt = st.slider("Voltage", 100.0, 300.0, 170.0)
                rotate = st.slider("Rotation", 300.0, 600.0, 450.0)
                pressure = st.slider("Pressure", 50.0, 200.0, 100.0)
                vibration = st.slider("Vibration", 10.0, 150.0, 35.0)
            with st.expander("Rolling Statistics (24h Mean/Std)", expanded=False):
                v_mean = st.number_input("Voltage Mean", value=volt)
                v_std = st.number_input("Voltage Std", value=10.0)
                r_mean = st.number_input("Rotation Mean", value=rotate)
                r_std = st.number_input("Rotation Std", value=50.0)
                p_mean = st.number_input("Pressure Mean", value=pressure)
                p_std = st.number_input("Pressure Std", value=10.0)
                vi_mean = st.number_input("Vibration Mean", value=vibration)
                vi_std = st.number_input("Vibration Std", value=5.0)
        with col2:
            st.subheader("Advanced Features")
            with st.expander("Lag Features (3h Changes)", expanded=True):
                v_change = st.number_input("Voltage Change (3h)", value=0.0)
                r_change = st.number_input("Rotation Change (3h)", value=0.0)
                p_change = st.number_input("Pressure Change (3h)", value=0.0)
                vi_change = st.number_input("Vibration Change (3h)", value=0.0)
            with st.expander("Maintenance Recency", expanded=True):
                d1 = st.number_input("Days since Component 1 replacement", value=10.0)
                d2 = st.number_input("Days since Component 2 replacement", value=10.0)
                d3 = st.number_input("Days since Component 3 replacement", value=10.0)
                d4 = st.number_input("Days since Component 4 replacement", value=10.0)
            if st.button("🚀 Predict Failure Probability", use_container_width=True):
                payload = {
                    "machine_id": machine_id, "volt": volt, "rotate": rotate, "pressure": pressure, "vibration": vibration,
                    "volt_mean_24h": v_mean, "volt_std_24h": v_std, "rotate_mean_24h": r_mean, "rotate_std_24h": r_std,
                    "pressure_mean_24h": p_mean, "pressure_std_24h": p_std, "vibration_mean_24h": vi_mean, "vibration_std_24h": vi_std,
                    "volt_lag_3h": volt - v_change, "volt_change_3h": v_change, "rotate_lag_3h": rotate - r_change,
                    "rotate_change_3h": r_change, "pressure_lag_3h": pressure - p_change, "pressure_change_3h": p_change,
                    "vibration_lag_3h": vibration - vi_change, "vibration_change_3h": vi_change,
                    "days_since_comp1": d1, "days_since_comp2": d2, "days_since_comp3": d3, "days_since_comp4": d4
                }
                make_prediction_api(api_url, payload)
    else:
        st.subheader("Raw JSON Input")
        json_input = st.text_area("Payload", value=json.dumps({"machine_id": 1, "volt": 170.0, "rotate": 450.0, "pressure": 100.0, "vibration": 35.0, "volt_mean_24h": 170.0, "volt_std_24h": 10.0, "days_since_comp1": 5.0}, indent=2), height=300)
        if st.button("🚀 Predict from JSON", use_container_width=True):
            try:
                payload = json.loads(json_input)
                make_prediction_api(api_url, payload)
            except json.JSONDecodeError:
                st.error("Invalid JSON format.")

with tab2:
    st.subheader("Batch Evaluation & Flagging")
    st.markdown("""
    Upload a CSV file containing sensor data and ground truth (target).
    The system will calculate Recall, ROC-AUC and **flag potential failures**.
    """)
    
    uploaded_file = st.file_uploader("Upload Test CSV", type="csv")
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded File: {uploaded_file.name} ({len(test_df)} rows)")
        
        if st.button("📊 Run Evaluation"):
            try:
                # 1. Run Predictions
                X_test = test_df[features_list]
                test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]
                test_df['prediction'] = model.predict(X_test)
                
                # 2. Calculate Metrics if 'target' exists
                if 'target' in test_df.columns:
                    rc = recall_score(test_df['target'], test_df['prediction'])
                    auc = roc_auc_score(test_df['target'], test_df['pred_prob'])
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Batch Recall", f"{rc:.4f}")
                    m2.metric("Batch ROC-AUC", f"{auc:.4f}")
                else:
                    st.warning("No 'target' column found. Metric calculation skipped.")
                
                # 3. Flagging & Highlighting
                st.subheader("Flagged Potential Failures")
                
                # Filter flagged rows
                flagged_df = test_df[test_df['prediction'] == 1].copy()
                
                if len(flagged_df) > 0:
                    st.error(f"🚨 Found {len(flagged_df)} potential failures!")
                    
                    # Style the dataframe to highlight flags
                    def highlight_failures(s):
                        return ['background-color: #ffcccc' if s.prediction == 1 else '' for _ in s]
                    
                    st.dataframe(flagged_df.head(100)) # Show top 100 flags
                else:
                    st.success("No failures predicted in this batch.")
                
                # Full Results with Download
                st.divider()
                st.subheader("Full Results")
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Full Results", data=csv, file_name="predictions.csv", mime='text/csv')
                st.write(test_df.head(10))
                
            except Exception as e:
                st.error(f"Evaluation Error: {str(e)}")
                st.info("Ensure the CSV has the correct feature columns.")

st.divider()
st.caption("Predictive Maintenance Pipeline - Google Deepmind Advanced Agentic Coding")
