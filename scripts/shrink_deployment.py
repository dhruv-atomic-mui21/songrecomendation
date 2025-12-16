
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from src.preprocessing import run_preprocessing_pipeline
from src.features import FeatureEngine
from src import config
import joblib

def shrink_data():
    print("Shrinking data for deployment...")
    
    # 1. Load full data
    df = pd.read_csv('data/data.csv')
    
    # 2. Extract Metadata (Display cols)
    meta_cols = ['id', 'name', 'artists', 'popularity', 'year']
    df_small = df[meta_cols].copy()
    
    # Save small CSV
    df_small.to_csv('data/data_small.csv', index=False)
    print(f"Saved data/data_small.csv. Size: {os.path.getsize('data/data_small.csv') / 1024 / 1024:.2f} MB")
    
    # 3. Pre-compute Features (so we don't need raw data to compute them)
    # We need to run the pipeline one last time on full data
    processed = run_preprocessing_pipeline('data/data.csv')
    engine = FeatureEngine(processed)
    features, _ = engine.extract_baseline_features()
    
    # Convert to float32 to save space
    features_32 = features.astype(np.float32)
    
    # Save features artifact
    joblib.dump(features_32, 'artifacts/features.joblib', compress=3)
    print(f"Saved artifacts/features.joblib. Size: {os.path.getsize('artifacts/features.joblib') / 1024 / 1024:.2f} MB")
    
    # 4. Retrain KNN with float32 to shrink model
    from sklearn.neighbors import NearestNeighbors
    print("Retraining KNN with float32...")
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
    knn.fit(features_32)
    joblib.dump(knn, config.KNN_MODEL_PATH, compress=3)
    print(f"Saved artifacts/models/knn_model.joblib. Size: {os.path.getsize(config.KNN_MODEL_PATH) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    shrink_data()
