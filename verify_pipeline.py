
import pandas as pd
import numpy as np
import os
import joblib
from src.preprocessing import run_preprocessing_pipeline

def verify_pipeline():
    print("Verifying Preprocessing Pipeline...")
    
    # 1. Load Data
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # 2. Run Pipeline
    processed_data = run_preprocessing_pipeline(data_path)
    print(f"Processed shape: {processed_data.shape}")
    
    # 3. Validation Check
    # Check for missing values
    if np.isnan(processed_data).any():
         print("Error: NaN values found in processed data.")
    else:
         print("Success: No NaN values.")
         
    # Check for scaler artifact
    scaler_path = 'artifacts/scalers/preprocessing_pipeline.joblib'
    if os.path.exists(scaler_path):
        print(f"Success: Artifact found at {scaler_path}")
        
    else:
        print(f"Error: Artifact not found at {scaler_path}")

if __name__ == "__main__":
    verify_pipeline()
