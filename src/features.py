
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureEngine:
    def __init__(self, processed_data):
        """
        Initialize with processed data (pandas DataFrame).
        """
        self.data = processed_data
        self.features = None
        self.popularity = None
        self.pca = None
        
    def extract_baseline_features(self):
        """
        Separates 'popularity' from the rest of the features.
        """
        # Identify popularity column (assuming it's named 'popularity')
        if 'popularity' in self.data.columns:
            self.popularity = self.data['popularity'].copy()
            self.features = self.data.drop(columns=['popularity']).copy()
        else:
            # Fallback if popularity was dropped or renamed, though it shouldn't be based on preprocessing
            print("Warning: 'popularity' column not found. Using all columns as features.")
            self.features = self.data.copy()
            self.popularity = pd.Series(np.zeros(len(self.data)), index=self.data.index)

        return self.features, self.popularity

    def apply_pca(self, n_components=None):
        """
        Applies PCA to the feature vector.
        """
        if self.features is None:
            raise ValueError("Run extract_baseline_features first.")
            
        print(f"Applying PCA with n_components={n_components}...")
        self.pca = PCA(n_components=n_components)
        reduced_data = self.pca.fit_transform(self.features)
        
        # Convert back to DataFrame for easier handling
        col_names = [f'pca_{i}' for i in range(reduced_data.shape[1])]
        self.features = pd.DataFrame(reduced_data, columns=col_names, index=self.features.index)
        
        return self.features

    def apply_feature_weighting(self, weights_dict):
        """
        Applies weights to specific features. 
        Note: This works best BEFORE PCA, but user request implies iterative extensions.
        If PCA is already applied, this won't work on original features.
        So this should be called *before* PCA if weighting original features.
        """
        if self.features is None:
             raise ValueError("Run extract_baseline_features first.")
        
        # Check if we are in PCA space
        if 'pca_0' in self.features.columns:
            print("Warning: Applying feature weighting after PCA is not recommended for original features.")
            return self.features

        print(f"Applying feature weighting: {weights_dict}...")
        for feature, weight in weights_dict.items():
            if feature in self.features.columns:
                self.features[feature] = self.features[feature] * weight
            else:
                 print(f"Warning: Feature '{feature}' not found in data.")
                 
        return self.features

    def apply_popularity_damping(self):
        """
        Damps the popularity feature using log1p.
        """
        if self.popularity is None:
             raise ValueError("Run extract_baseline_features first.")
        
        self.popularity = 1 / (1 + np.exp(-self.popularity))
        return self.popularity
