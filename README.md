
# Spotify Song Recommender

This project implements a content-based song recommender system using the Spotify dataset.

## Structure

-   `data/`: Contains raw data.
-   `src/`: Source code for preprocessing, feature engineering, modeling, and recommendation.
-   `notebooks/`: Jupyter notebooks for analysis and experiments.
-   `artifacts/`: Saved models and scalers.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Ensure data is in `data/data.csv`.

## Usage

### Recommendation Service

```python
from src.recommend import RecommendationService

service = RecommendationService(model_strategy='knn')
recommendations = service.get_recommendations_by_name("Stardust", k=5)
print(recommendations)
```

## Pipeline

1.  **Preprocessing**: `src/preprocessing.py` cleans and encodes data.
2.  **Features**: `src/features.py` handles weighting and PCA.
3.  **Models**: `src/models.py` implements Cosine, KNN, and Embedding strategies.
4.  **Recommend**: `src/recommend.py` provides the main API.
