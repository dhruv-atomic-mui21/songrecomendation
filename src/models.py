
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import config

class Recommender:
    def __init__(self, features_df, strategy='cosine'):
        """
        Initialize Recommender.
        
        Args:
            features_df (pd.DataFrame): The feature vectors (preprocessed).
            strategy (str): 'cosine', 'knn', or 'embedding'.
        """
        self.features_df = features_df
        self.strategy = strategy
        self.model = None
        self.pca = None
        self.embeddings = None
        
    def train(self, n_neighbors=10, n_components=50):
        """
        Trains the model based on the strategy.
        """
        print(f"Training Recommender with strategy: {self.strategy}")
        
        if self.strategy == 'cosine':
            # No training needed for on-the-fly cosine
            pass
            
        elif self.strategy == 'knn':
            self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
            self.model.fit(self.features_df)
            # Save model
            joblib.dump(self.model, config.KNN_MODEL_PATH)
            
        elif self.strategy == 'embedding':
            # Dynamically adjust n_components if it exceeds feature count
            n_features = self.features_df.shape[1]
            if n_components > n_features:
                print(f"Reduction components adjusted from {n_components} to {n_features} (max available).")
                n_components = n_features

            # Train PCA
            print(f"Fitting PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components)
            self.embeddings = self.pca.fit_transform(self.features_df)
            
            # Save artifacts
            joblib.dump(self.pca, config.PCA_MODEL_PATH)
            np.save(config.EMBEDDINGS_PATH, self.embeddings)
            print("Embeddings and PCA model saved.")
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def recommend(self, item_index, k=10):
        """
        Recommends k items similar to item_index.
        Returns: list of indices, list of scores (distances/similarities)
        """
        if item_index >= len(self.features_df):
            raise IndexError("Item index out of bounds")

        # Keep as DataFrame to preserve feature names for KNN
        target_vector = self.features_df.iloc[[item_index]]

        if self.strategy == 'cosine':
            # Calculate cosine similarity against all items
            # Note: This computes similarity (higher is better)
            sim_scores = cosine_similarity(target_vector, self.features_df).flatten()
            
            # Get top k (excluding self)
            # argpartition is faster than sort for top k
            # We want top k+1 because self is included
            top_indices = np.argpartition(sim_scores, -(k+1))[-(k+1):]
            
            # Sort these top indices by score descending
            top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
            
            # Remove self if present (it should be first)
            top_indices = top_indices[top_indices != item_index][:k]
            scores = sim_scores[top_indices]
            
            return top_indices, scores

        elif self.strategy == 'knn':
            if self.model is None:
                raise RuntimeError("Model not trained. Call train() first.")
                
            # kneighbors returns distances (lower is better for distance metrics)
            # Since we used metric='cosine', distance = 1 - similarity
            distances, indices = self.model.kneighbors(target_vector, n_neighbors=k+1)
            
            # Flatten
            indices = indices.flatten()
            distances = distances.flatten()
            
            # Exclude self (first item)
            return indices[1:], distances[1:]

        elif self.strategy == 'embedding':
            print("Embedding strategy mostly for artifact generation in this environment.")
            print("Returning placeholder (neighbors not computed). Check artifacts.")
            return [], []
            
        return [], []
