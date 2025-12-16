
import pandas as pd
import numpy as np
from src.preprocessing import run_preprocessing_pipeline
from src.features import FeatureEngine
from src.models import Recommender
import time

def verify_models():
    print("Verifying Model Progression...")
    
    # 1. Prepare Data
    # Assuming data exists
    features_df = run_preprocessing_pipeline('data/data.csv')
    
    # Feature Engineering (just baseline for now to save time/complexity)
    engine = FeatureEngine(features_df)
    features, _ = engine.extract_baseline_features()
    
    # Work with a subset for speed in verification if dataset is huge, 
    # but let's try full set single query first.
    print(f"Data shape: {features.shape}")
    
    test_idx = 0
    
    # 2. Stage 1: Cosine
    print("\n--- Stage 1: Cosine Similarity ---")
    start = time.time()
    rec_cosine = Recommender(features, strategy='cosine')
    rec_cosine.train()
    indices_cos, scores_cos = rec_cosine.recommend(test_idx, k=5)
    end = time.time()
    print(f"Cosine Recommendation time: {end - start:.4f}s")
    print(f"Indices: {indices_cos}")
    print(f"Scores: {scores_cos}")
    
    # 3. Stage 2: KNN
    print("\n--- Stage 2: KNN ---")
    start = time.time()
    rec_knn = Recommender(features, strategy='knn')
    rec_knn.train(n_neighbors=5)
    indices_knn, scores_knn = rec_knn.recommend(test_idx, k=5)
    end = time.time()
    print(f"KNN Recommendation time (query): {end - start:.4f}s")
    print(f"Indices: {indices_knn}")
    
    # Consistency Check
    # KNN with cosine metric returns Distances. Cosine Sim returns Similarity.
    # Distance = 1 - Similarity roughly (or angle).
    # Indices should match strongly.
    intersection = len(set(indices_cos).intersection(set(indices_knn)))
    print(f"Intersection between Cosine and KNN indices: {intersection}/5")
    
    # 4. Stage 3: Embedding
    print("\n--- Stage 3: Embedding Artifacts ---")
    rec_emb = Recommender(features, strategy='embedding')
    rec_emb.train(n_components=10) # Small n for speed
    
    import os
    if os.path.exists('artifacts/embeddings/embeddings.npy') and \
       os.path.exists('artifacts/models/pca_model.joblib'):
        print("Success: Embedding artifacts found.")
    else:
        print("Error: Embedding artifacts missing.")

if __name__ == "__main__":
    verify_models()
