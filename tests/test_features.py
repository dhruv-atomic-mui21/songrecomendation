
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.features import FeatureEngine
from src.preprocessing import run_preprocessing_pipeline

@pytest.fixture(scope="module")
def processed_data():
    if not os.path.exists('data/data.csv'):
        # Mock data if file missing
        df = pd.DataFrame(np.random.rand(100, 15), columns=[f'col_{i}' for i in range(14)] + ['popularity'])
        return df
    return run_preprocessing_pipeline('data/data.csv')

def test_baseline_extraction(processed_data):
    engine = FeatureEngine(processed_data)
    features, pop = engine.extract_baseline_features()
    
    assert 'popularity' not in features.columns
    assert len(pop) == len(features)
    assert not features.empty

def test_feature_weighting(processed_data):
    engine = FeatureEngine(processed_data)
    features, _ = engine.extract_baseline_features()
    
    # Pick a feature that exists
    feature_name = features.columns[0]
    original_val = features[feature_name].iloc[0]
    
    engine.apply_feature_weighting({feature_name: 2.0})
    new_val = engine.features[feature_name].iloc[0]
    
    # Allow float precision
    assert abs(new_val - (original_val * 2.0)) < 1e-5

def test_pca_reduction(processed_data):
    engine = FeatureEngine(processed_data)
    engine.extract_baseline_features()
    
    n_components = 5
    # Ensure n_components < n_features
    if processed_data.shape[1] > n_components:
        engine.apply_pca(n_components=n_components)
        assert engine.features.shape[1] == n_components
    else:
        pytest.skip("Not enough features for PCA test")

def test_popularity_damping(processed_data):
    engine = FeatureEngine(processed_data)
    engine.extract_baseline_features()
    
    pop = engine.apply_popularity_damping()
    assert pop.min() >= 0
    assert pop.max() <= 1.0 # sigmoid output
