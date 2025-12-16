
import pandas as pd
import os
import config

def load_data(path=None):
    """
    Loads data from CSV.
    """
    if path is None:
        path = config.RAW_DATA_PATH
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
        
    return pd.read_csv(path)
