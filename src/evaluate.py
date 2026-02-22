import pandas as pd
import joblib
import os
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix

def evaluate_model(data_path, model_path, features_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'target' not in df.columns:
        print("Error: Dataset must contain a 'target' column (ground truth) for evaluation.")
        return

    print(f"Loading model and features...")
    model = joblib.load(model_path)
    features_list = joblib.load(features_path)
    
    X = df[features_list]
    y_true = df['target']
    
    print("Generating predictions...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    
    print("\n" + "="*30)
    print(" EVALUATION RESULTS ")
    print("="*30)
    for m, v in metrics.items():
        print(f"{m:12}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"True Positives  : {tp}")
    print(f"False Negatives : {fn} (Missed failures)")
    print(f"False Positives : {fp} (False alarms)")
    print(f"True Negatives  : {tn}")
    print("="*30)

if __name__ == "__main__":
    # Example usage: evaluates on the existing test (processed) data
    # In a real case, you'd replace this with a path to a NEW labeled dataset
    DATA_PATH = r'd:\Jitendra\Data\processed_data.csv' 
    MODEL_PATH = 'src/model.joblib'
    FEATURES_PATH = 'src/features_list.joblib'
    
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        evaluate_model(DATA_PATH, MODEL_PATH, FEATURES_PATH)
    else:
        print("Required files not found. Ensure you have run preprocess.py and train.py first.")
