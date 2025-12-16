
import pandas as pd
import numpy as np
import os
import joblib
from preprocessing import run_preprocessing_pipeline
from features import FeatureEngine
from models import Recommender
import config, data_loader

class RecommendationService:
    def __init__(self, data_path=None, model_strategy='knn'):
        """
        Initialize RecommendationService.
        
        Args:
            data_path (str): Path to raw data CSV.
            model_strategy (str): 'cosine', 'knn', or 'embedding'.
        """
        self.strategy = model_strategy
        self.data_path = data_path
        
        # Load Optimized Metadata (Deployment Mode)
        if os.path.exists('data/data_small.csv'):
            print("Loading optimized metadata...")
            self.raw_data = pd.read_csv('data/data_small.csv')
        else:
            print("Loading raw data...")
            self.raw_data = data_loader.load_data(data_path)
            
        # Load Optimized Features (Deployment Mode)
        if os.path.exists('artifacts/features.joblib'):
            print("Loading pre-computed features...")
            self.features = joblib.load('artifacts/features.joblib')
            # Verify alignment if possible, assuming index matches 
            # (reset_index if needed, but data_small was sliced from data so index 0..N should match)
        else:
            # Fallback for Development/Training
            print("Running preprocessing pipeline (slow)...")
            features_df = run_preprocessing_pipeline(data_path)
            
            # Feature Engineering
            self.feature_engine = FeatureEngine(features_df)
            self.features, self.popularity = self.feature_engine.extract_baseline_features()
        
        # Initialize Recommender
        self.recommender = Recommender(self.features, strategy=self.strategy)
        # Check if model exists to avoid re-training in prod if strategy is knn
        if self.strategy == 'knn' and os.path.exists(config.KNN_MODEL_PATH):
             print(f"Loading existing model from {config.KNN_MODEL_PATH}")
             self.recommender.model = joblib.load(config.KNN_MODEL_PATH)
        else:
             self.recommender.train()

    def get_recommendations_by_name(self, track_name, k=10):
        """
        Get recommendations by track name.
        """
        # Find index
        # Simple case-insensitive exact match first
        matches = self.raw_data[self.raw_data['name'].str.lower() == track_name.lower()]
        
        if matches.empty:
            # Try partial match
            matches = self.raw_data[self.raw_data['name'].str.lower().str.contains(track_name.lower())]
            
        if matches.empty:
            return {"error": f"Track '{track_name}' not found."}
            
        # If multiple, take the most popular one
        best_match_idx = matches['popularity'].idxmax()
        actual_name = self.raw_data.loc[best_match_idx, 'name']
        artist = self.raw_data.loc[best_match_idx, 'artists']
        print(f"Found match: '{actual_name}' by {artist} (Index: {best_match_idx})")
        
        return self.get_recommendations_by_index(best_match_idx, k=k)

    def get_recommendations_by_index(self, index, k=10):
        """
        Get recommendations by index.
        """
        try:
            indices, scores = self.recommender.recommend(index, k=k)
        except Exception as e:
            return {"error": str(e)}
            
        results = []
        for i, idx in enumerate(indices):
            track_info = self.raw_data.iloc[idx].to_dict()
            distance = float(scores[i]) if len(scores) > i else 0.0
            
            # Add metadata
            track_info['distance_score'] = distance
            track_info['similarity_score'] = 1.0 - distance
            track_info['rank'] = i + 1
            
            # Clean up numpy types for JSON serialization if needed
            # (pandas usually handles this well in to_dict, but being safe)
            results.append(track_info)
            
        return {
            "query_track": self.raw_data.iloc[index].to_dict(),
            "recommendations": results
        }

if __name__ == "__main__":
    # Example usage
    service = RecommendationService(model_strategy='knn')
    rec = service.get_recommendations_by_name("Stardust", k=5)
    import pprint
    pprint.pprint(rec)
