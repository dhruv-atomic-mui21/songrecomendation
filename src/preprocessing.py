
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import config, data_loader

def run_preprocessing_pipeline(data_path=None):
    """
    Reads data, runs the preprocessing pipeline, and saves the fitted pipeline.
    Returns the processed data (pandas DataFrame).
    """
    # 1. Load Data
    try:
        df = data_loader.load_data(data_path)
    except FileNotFoundError as e:
        raise e
    
    # 2. Define Features
    # Numeric features to scale
    numeric_features = [
        'valence', 'year', 'acousticness', 'danceability', 
        'duration_ms', 'energy', 'instrumentalness', 'liveness', 
        'loudness', 'popularity', 'speechiness', 'tempo'
    ]
    
    # Categorical features to encode
    categorical_features = ['key', 'mode', 'explicit']
    
    # Columns to drop are implicitly handled by remainder='drop'
    
    # 3. Define Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Create Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',  # This drops 'id', 'name', 'artists', 'release_date'
        verbose_feature_names_out=False # cleaner column names
    )
    
    # 5. Create Final Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Enable pandas output
    pipeline.set_output(transform='pandas')
    
    # 6. Fit and Transform
    print("Fitting pipeline...")
    processed_data = pipeline.fit_transform(df)
    
    # 7. Persist Pipeline
    joblib.dump(pipeline, config.PIPELINE_PATH)
    print(f"Pipeline saved to {config.PIPELINE_PATH}")
    
    return processed_data

if __name__ == "__main__":
    # Example usage
    try:
        data = run_preprocessing_pipeline()
        print(f"Preprocessing complete. Data shape: {data.shape}")
    except Exception as e:
        print(f"Error: {e}")
