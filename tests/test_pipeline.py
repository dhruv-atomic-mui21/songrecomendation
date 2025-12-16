
import pytest
import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from src.preprocessing import run_preprocessing_pipeline
from src import config

@pytest.fixture(scope="module")
def data_setup():
    # Setup: Ensure artifacts dir exists
    os.makedirs(os.path.dirname(config.PIPELINE_PATH), exist_ok=True)
    yield
    # Teardown (optional): clean up artifacts if strictly unit testing, 
    # but here we might want to keep them or mocking might be better.
    # For integration/deployment tests, using real files is fine.

def test_pipeline_execution(data_setup):
    """Test full pipeline execution."""
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        pytest.skip(f"{data_path} not found")

    processed_data = run_preprocessing_pipeline(data_path)
    
    assert not processed_data.empty, "Processed dataframe is empty"
    assert not np.isnan(processed_data.values).any(), "NaN values found in processed data"
    
def test_pipeline_artifacts(data_setup):
    """Test artifact creation."""
    assert os.path.exists(config.PIPELINE_PATH), "Pipeline artifact not saved"
