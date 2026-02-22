import pandas as pd
import numpy as np
import os

def load_data(data_dir):
    telemetry = pd.read_csv(os.path.join(data_dir, 'telemetry_v2.csv'))
    maintenance = pd.read_csv(os.path.join(data_dir, 'maintenance_v2.csv'))
    failures = pd.read_csv(os.path.join(data_dir, 'failures_v2.csv'))
    
    # Convert timestamps to datetime
    telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
    maintenance['timestamp'] = pd.to_datetime(maintenance['timestamp'])
    failures['timestamp'] = pd.to_datetime(failures['timestamp'])
    
    return telemetry, maintenance, failures

def preprocess_telemetry(telemetry):
    # Sort by machine and timestamp
    telemetry = telemetry.sort_values(['machine_id', 'timestamp'])
    
    # Handle missing values - forward fill then backfill within each machine group
    telemetry = telemetry.groupby('machine_id').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    # 1. Rolling Statistics (24-hour mean/std)
    sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
    for col in sensor_cols:
        telemetry[f'{col}_mean_24h'] = telemetry.groupby('machine_id')[col].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
        telemetry[f'{col}_std_24h'] = telemetry.groupby('machine_id')[col].transform(lambda x: x.rolling(window=24, min_periods=1).std())
    
    # 2. Lag Features (3-hour change)
    for col in sensor_cols:
        telemetry[f'{col}_lag_3h'] = telemetry.groupby('machine_id')[col].shift(3)
        telemetry[f'{col}_change_3h'] = telemetry[col] - telemetry[f'{col}_lag_3h']
    
    return telemetry

def engineer_recency(telemetry, maintenance):
    # 3. Recency Features (Days since last replacement)
    # Pivot maintenance to get timestamps for each component replacement
    maint_pivot = maintenance.pivot_table(index=['timestamp', 'machine_id'], columns='component', aggfunc='size', fill_value=0).reset_index()
    
    # Merge with telemetry
    df = pd.merge(telemetry, maint_pivot, on=['timestamp', 'machine_id'], how='left').fillna(0)
    
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        # Create a helper column that is the timestamp if that component was replaced, else NaT
        df.loc[df[comp] == 1, f'{comp}_last'] = df['timestamp']
        # Forward fill the last replacement date
        df[f'{comp}_last'] = df.groupby('machine_id')[f'{comp}_last'].ffill()
        # Calculate days since last replacement
        df[f'days_since_{comp}'] = (df['timestamp'] - df[f'{comp}_last']).dt.total_seconds() / (24 * 3600)
        # Drop helper column
        df.drop(columns=[f'{comp}_last', comp], inplace=True)
        
    return df

def label_failures(df, failures, window=24):
    # Match failures with telemetry
    # For each failure, mark the previous 'window' hours as failure=1
    df['target'] = 0
    
    for _, row in failures.iterrows():
        machine_id = row['machine_id']
        failure_time = row['timestamp']
        
        # Define window: [failure_time - window_hours, failure_time]
        start_time = failure_time - pd.Timedelta(hours=window)
        
        # Update target for this machine in this time window
        mask = (df['machine_id'] == machine_id) & \
               (df['timestamp'] > start_time) & \
               (df['timestamp'] <= failure_time)
        df.loc[mask, 'target'] = 1
        
    return df

if __name__ == "__main__":
    DATA_DIR = r'd:\Jitendra\Data'
    telemetry, maintenance, failures = load_data(DATA_DIR)
    
    print("Preprocessing telemetry...")
    df = preprocess_telemetry(telemetry)
    
    print("Engineering recency features...")
    df = engineer_recency(df, maintenance)
    
    print("Labeling failures...")
    df = label_failures(df, failures)
    
    # Save the processed data
    output_path = os.path.join(DATA_DIR, 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(df.head())
    print(df.columns)
