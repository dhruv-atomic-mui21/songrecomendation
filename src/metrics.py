import numpy as np

def calculate_average_similarity(scores):
    """
    Calculates average similarity score from a list of scores.
    """
    if not scores:
        return 0.0
    return np.mean(scores)

def precision_at_k(recommended_indices, ground_truth_indices, k):
    """
    Calculates Precision@K. 
    Note: Requires ground truth which is usually not available in unsupervised recommendation.
    Included for structure completeness.
    """
    k = min(k, len(recommended_indices))
    recommended_k = set(recommended_indices[:k])
    ground_truth_set = set(ground_truth_indices)
    
    intersection = recommended_k.intersection(ground_truth_set)
    return len(intersection) / k
