
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'data.csv')

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
SCALERS_DIR = os.path.join(ARTIFACTS_DIR, 'scalers')
MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'models')
EMBEDDINGS_DIR = os.path.join(ARTIFACTS_DIR, 'embeddings')

# Ensure directories exist
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Files
PIPELINE_PATH = os.path.join(SCALERS_DIR, 'preprocessing_pipeline.joblib')
KNN_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.joblib')
PCA_MODEL_PATH = os.path.join(MODELS_DIR, 'pca_model.joblib')
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, 'embeddings.npy')

# Model Config
RANDOM_STATE = 42
PCA_COMPONENTS = 50
KNN_NEIGHBORS = 10
