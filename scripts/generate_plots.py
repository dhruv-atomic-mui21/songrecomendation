
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is importable
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.preprocessing import run_preprocessing_pipeline
from src.features import FeatureEngine

def generate_visualizations():
    print("Generating visualizations...")
    os.makedirs('assets/images', exist_ok=True)
    
    # 1. Load Data
    features_df = run_preprocessing_pipeline('data/data.csv')
    engine = FeatureEngine(features_df)
    features, _ = engine.extract_baseline_features()
    
    # 2. PCA Variance Plot
    print("Generating PCA Explained Variance plot...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=25)
    pca.fit(features)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 26), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance')
    plt.legend()
    plt.savefig('assets/images/pca_variance.png')
    plt.close()
    print("Saved assets/images/pca_variance.png")
    
    # 3. Feature Distribution / Correlation (Optional but good for showcase)
    # Let's do a simple correlation heatmap of numeric features
    print("Generating Correlation Heatmap...")
    numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    # Needs matching numeric columns from features (which might be scaled/encoded)
    # The clean numeric names should be available if we subset
    import seaborn as sns # Optional dependency, might fail if not installed. Fallback to matplotlib matshow
    
    try:
        # Check if columns exist (names might vary slightly after scaling but usually preserved if verbose_out=False)
        corr = features[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=45)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('assets/images/correlation_matrix.png')
        plt.close()
        print("Saved assets/images/correlation_matrix.png")
    except Exception as e:
        print(f"Skipping correlation plot: {e}")

if __name__ == "__main__":
    generate_visualizations()
