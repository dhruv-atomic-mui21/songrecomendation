
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.models import Recommender
from src.recommend import RecommendationService

@pytest.fixture
def mock_features():
    # 20 samples, 10 features
    return pd.DataFrame(np.random.rand(20, 10), columns=[f'f{i}' for i in range(10)])

def test_recommender_cosine(mock_features):
    rec = Recommender(mock_features, strategy='cosine')
    # No train needed
    indices, scores = rec.recommend(0, k=5)
    
    assert len(indices) == 5
    assert len(scores) == 5
    assert 0 not in indices # Should exclude self

def test_recommender_knn(mock_features):
    rec = Recommender(mock_features, strategy='knn')
    rec.train(n_neighbors=5)
    indices, scores = rec.recommend(0, k=5)
    
    assert len(indices) == 5
    assert len(scores) == 5
    
def test_recommender_embedding_dynamic_adjustment(mock_features):
    # Request 50 components, but only 10 features available
    rec = Recommender(mock_features, strategy='embedding')
    # Should not crash
    rec.train(n_components=50) 
    assert rec.pca.n_components_ == 10

def test_recommendation_service_integration():
    """Integration test for full service."""
    if not os.path.exists('data/data.csv'):
        pytest.skip("Data file missing")
        
    service = RecommendationService(model_strategy='knn')
    
    # Test valid song (using one from head of data or known name if possible)
    # Using index is safer for generic test
    res = service.get_recommendations_by_index(0, k=3)
    
    assert 'error' not in res
    assert len(res['recommendations']) == 3
    assert 'rank' in res['recommendations'][0]
    assert 'similarity_score' in res['recommendations'][0]
