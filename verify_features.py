
import pandas as pd
from src.preprocessing import run_preprocessing_pipeline
from src.features import FeatureEngine

def verify_features():
    print("Verifying Feature Engineering...")
    
    # 1. Get Preprocessed Data
    # Use existing artifact if possible to save time, but run_preprocessing_pipeline handles it fast
    try:
        data = run_preprocessing_pipeline('data/data.csv')
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    engine = FeatureEngine(data)
    
    # 2. Baseline extraction
    print("\n--- Baseline Extraction ---")
    features, pop = engine.extract_baseline_features()
    print(f"Features shape: {features.shape}")
    print(f"Popularity shape: {pop.shape}")
    if 'popularity' in features.columns:
        print("Error: Popularity still in features.")
    else:
        print("Success: Popularity separated.")

    # 3. Feature Weighting
    print("\n--- Feature Weighting ---")
    # Store old energy to compare
    if 'energy' in features.columns:
        old_energy = features['energy'].iloc[0]
        engine.apply_feature_weighting({'energy': 2.0, 'valence': 1.5})
        new_energy = engine.features['energy'].iloc[0]
        print(f"Old Energy: {old_energy}, New Energy: {new_energy}")
        if abs(new_energy - (old_energy * 2.0)) < 1e-5:
             print("Success: Weighting applied.")
        else:
             print("Error: Weighting mismatch.")
    else:
        print("Energy column not found (maybe PCA ran first?).")

    # 4. PCA
    print("\n--- PCA ---")
    engine.apply_pca(n_components=5)
    print(f"Reduced features shape: {engine.features.shape}")
    if engine.features.shape[1] == 5:
        print("Success: PCA reduced dimensions.")
    else:
        print("Error: Incorrect dimensions.")

    # 5. Popularity Damping
    print("\n--- Popularity Damping ---")
    print(f"Old Pop (Sample): {pop.iloc[0]}")
    damped_pop = engine.apply_popularity_damping()
    print(f"Damped Pop (Sample): {damped_pop.iloc[0]}")
    # Check bounds
    if damped_pop.min() >= 0 and damped_pop.max() <= 1:
        print("Success: Damped popularity in [0, 1].")
    else:
        print("Warning: Damped popularity out of expected bounds.")

if __name__ == "__main__":
    verify_features()
