import pandas as pd
import os

processed_data_path = r'd:\Jitendra\Data\processed_data.csv'
test_sample_path = r'd:\Jitendra\test_sample.csv'

if os.path.exists(processed_data_path):
    df = pd.read_csv(processed_data_path)
    
    # Take a sample that includes both failures and normal cases
    failures = df[df['target'] == 1].head(100)
    normal = df[df['target'] == 0].head(900)
    
    sample = pd.concat([failures, normal]).sample(frac=1).reset_index(drop=True)
    
    sample.to_csv(test_sample_path, index=False)
    print(f"Sample test file created at {test_sample_path} with {len(sample)} rows.")
else:
    print("Processed data not found. Run preprocess.py first.")
