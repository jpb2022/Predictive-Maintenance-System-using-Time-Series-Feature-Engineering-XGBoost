import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

def load_processed_data(data_path):
    df = pd.read_csv(data_path)
    # Drop non-feature columns
    drop_cols = ['timestamp', 'machine_id', 'failure', 'target']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['target']
    return X, y, features

def train_and_evaluate(X_train, X_test, y_train, y_test, features):
    # Calculate scale_pos_weight for class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
    
    print(f"Class imbalance: Negatives={num_neg}, Positives={num_pos}, Weight={scale_pos_weight:.2f}")
    
    # Baseline Model: Random Forest
    print("\nTraining Baseline (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # XGBoost Model
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n--- Baseline (Random Forest) ---")
    print(classification_report(y_test, rf_preds))
    
    print("\n--- XGBoost ---")
    print(classification_report(y_test, xgb_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    
    return xgb, rf, features

def generate_report(y_test, xgb_preds, xgb_probs):
    metrics = {
        'Precision': precision_score(y_test, xgb_preds),
        'Recall': recall_score(y_test, xgb_preds),
        'F1-Score': f1_score(y_test, xgb_preds),
        'ROC-AUC': roc_auc_score(y_test, xgb_probs)
    }
    
    report = "# Model Evaluation Report\n\n"
    report += "## Performance Metrics (XGBoost)\n\n"
    report += "| Metric | Value |\n"
    report += "| --- | --- |\n"
    for m, v in metrics.items():
        report += f"| {m} | {v:.4f} |\n"
    
    report += "\n## Choice of Metrics\n"
    report += "For predictive maintenance, **Accuracy** is misleading because machine failures are rare (highly imbalanced dataset). "
    report += "We prioritize **Recall** and **F1-Score**:\n"
    report += "- **Recall**: Ensures we catch as many failures as possible (minimizing false negatives).\n"
    report += "- **Precision**: Ensures we don't trigger too many false alarms, which leads to unnecessary maintenance costs.\n"
    report += "- **F1-Score**: Provides a balanced view between Precision and Recall.\n"
    report += "- **ROC-AUC**: Evaluates the model's ability to distinguish between classes across different thresholds.\n"
    
    with open('report.md', 'w') as f:
        f.write(report)
    print("Report saved to report.md")

if __name__ == "__main__":
    DATA_PATH = r'd:\Jitendra\Data\processed_data.csv'
    X, y, features = load_processed_data(DATA_PATH)
    
    # Time-based split would be better, but simple split for now
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    xgb_model, rf_model, features = train_and_evaluate(X_train, X_test, y_train, y_test, features)
    
    # Save best model (XGBoost)
    joblib.dump(xgb_model, 'src/model.joblib')
    joblib.dump(features, 'src/features_list.joblib')
    print("Model and feature list saved to src/")
    
    # Generate report
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    generate_report(y_test, xgb_preds, xgb_probs)
