
import pprint
from src.recommend import RecommendationService

def verify_recommendation():
    print("Verifying Recommendation Service...")
    
    # 1. Initialize
    try:
        service = RecommendationService(model_strategy='knn')
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. Test Cases
    tracks_to_test = ["Stardust", "Bohemian Rhapsody", "Shape of You", "NonExistentSong123"]
    
    for track in tracks_to_test:
        print(f"\n--- Testing: {track} ---")
        result = service.get_recommendations_by_name(track, k=5)
        
        if "error" in result:
            print(f"Result: Error - {result['error']}")
        else:
            query = result['query_track']
            recs = result['recommendations']
            print(f"Query: {query['name']} - {query['artists']}")
            print(f"Top 1 Rec: {recs[0]['name']} - {recs[0]['artists']} (Score: {recs[0]['distance_score']:.4f})")
            
            # Validation
            if len(recs) == 5:
                print("Success: 5 recommendations returned.")
            else:
                print("Error: Incorrect number of recommendations.")

if __name__ == "__main__":
    verify_recommendation()
