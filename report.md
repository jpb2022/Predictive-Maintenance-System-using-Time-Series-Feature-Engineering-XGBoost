# Model Evaluation Report

## Performance Metrics (XGBoost)

| Metric | Value |
| --- | --- |
| Precision | 0.5718 |
| Recall | 0.9151 |
| F1-Score | 0.7039 |
| ROC-AUC | 0.9653 |

## Choice of Metrics
For predictive maintenance, **Accuracy** is misleading because machine failures are rare (highly imbalanced dataset). We prioritize **Recall** and **F1-Score**:
- **Recall**: Ensures we catch as many failures as possible (minimizing false negatives).
- **Precision**: Ensures we don't trigger too many false alarms, which leads to unnecessary maintenance costs.
- **F1-Score**: Provides a balanced view between Precision and Recall.
- **ROC-AUC**: Evaluates the model's ability to distinguish between classes across different thresholds.
