
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.recommend import RecommendationService

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Service (Load model once on startup)
# In production, you might want to lazy load or handle errors gracefully
try:
    print("Initializing Recommendation Service...")
    service = RecommendationService(model_strategy='knn')
    print("Service initialized successfully.")
except Exception as e:
    print(f"Failed to initialize service: {e}")
    service = None

@app.route('/')
@app.route('/frontend')
def frontend():
    """Renders the web interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if service else "unhealthy"
    return jsonify({"status": status}), 200 if service else 503

@app.route('/recommend', methods=['POST'])
def recommend():
    if not service:
        return jsonify({"error": "Service not initialized"}), 503

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Mode 1: Recommend by Song Name
    if 'song_name' in data:
        k = data.get('k', 10)
        song_name = data['song_name']
        print(f"Request: Recommend for '{song_name}' (k={k})")
        
        result = service.get_recommendations_by_name(song_name, k=k)
        
        if "error" in result:
             return jsonify(result), 404 # Not found
        
        return jsonify(result)

    # Mode 2: Recommend by Index (for internal testing mostly)
    elif 'song_index' in data:
        k = data.get('k', 10)
        idx = data['song_index']
        result = service.get_recommendations_by_index(idx, k=k)
        if "error" in result:
             return jsonify(result), 400
        return jsonify(result)

    else:
        return jsonify({"error": "Please provide 'song_name' or 'song_index' in request body"}), 400

if __name__ == '__main__':
    # Threaded=True for simple concurrent handling in dev
    app.run(host='0.0.0.0', port=5000, debug=False)
