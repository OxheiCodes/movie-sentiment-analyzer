from flask import Flask, request, jsonify
from flask_cors import CORS
from model import SentimentAnalyzer
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_REVIEW_LENGTH'] = int(os.getenv('MAX_REVIEW_LENGTH', 5000))
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'models')

# Initialize sentiment analyzer
print("Loading sentiment analysis model...")
try:
    analyzer = SentimentAnalyzer(model_path=app.config['MODEL_PATH'])
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    analyzer = None

# Request counter for monitoring
request_count = 0

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Movie Sentiment Analyzer API',
        'version': '1.0.0',
        'model_loaded': analyzer is not None,
        'endpoints': {
            'predict': '/api/predict',
            'health': '/api/health',
            'stats': '/api/stats'
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer is not None,
        'total_requests': request_count,
        'timestamp': time.time()
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    if not analyzer:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Load model metadata
    import json
    metadata_path = os.path.join(app.config['MODEL_PATH'], 'model_metadata.json')
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {}
    
    return jsonify({
        'total_requests': request_count,
        'model_info': metadata,
        'max_review_length': app.config['MAX_REVIEW_LENGTH']
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for a movie review
    
    Expected JSON body:
    {
        "review": "Your movie review text here"
    }
    
    Returns:
    {
        "sentiment": "Positive" or "Negative",
        "confidence": 85.23,
        "prediction_time_ms": 2.45,
        "label": 1
    }
    """
    global request_count
    request_count += 1
    
    # Check if model is loaded
    if not analyzer:
        return jsonify({
            'error': 'Model not available',
            'message': 'Sentiment analysis model failed to load'
        }), 503
    
    # Validate request has JSON body
    if not request.is_json:
        return jsonify({
            'error': 'Invalid request format',
            'message': 'Request must be JSON with Content-Type: application/json'
        }), 400
    
    # Get request data
    data = request.get_json()
    
    # Validate review field exists
    if 'review' not in data:
        return jsonify({
            'error': 'Missing required field',
            'message': 'Request body must contain "review" field'
        }), 400
    
    review_text = data['review']
    
    # Validate review is a string
    if not isinstance(review_text, str):
        return jsonify({
            'error': 'Invalid review format',
            'message': 'Review must be a string'
        }), 400
    
    # Validate review is not empty
    if not review_text.strip():
        return jsonify({
            'error': 'Empty review',
            'message': 'Review text cannot be empty'
        }), 400
    
    # Validate review length
    if len(review_text) > app.config['MAX_REVIEW_LENGTH']:
        return jsonify({
            'error': 'Review too long',
            'message': f'Review must be less than {app.config["MAX_REVIEW_LENGTH"]} characters',
            'current_length': len(review_text)
        }), 400
    
    # Make prediction
    try:
        result = analyzer.predict(review_text)
        
        # Check if prediction had an error
        if 'error' in result:
            return jsonify(result), 500
        
        # Add request metadata
        result['review_length'] = len(review_text)
        result['request_id'] = request_count
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple reviews
    
    Expected JSON body:
    {
        "reviews": ["Review 1", "Review 2", ...]
    }
    """
    global request_count
    
    if not analyzer:
        return jsonify({'error': 'Model not available'}), 503
    
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
    
    data = request.get_json()
    
    if 'reviews' not in data:
        return jsonify({'error': 'Missing required field "reviews"'}), 400
    
    reviews = data['reviews']
    
    if not isinstance(reviews, list):
        return jsonify({'error': 'Reviews must be a list'}), 400
    
    if len(reviews) == 0:
        return jsonify({'error': 'Reviews list cannot be empty'}), 400
    
    if len(reviews) > 100:
        return jsonify({'error': 'Maximum 100 reviews per batch request'}), 400
    
    # Process each review
    results = []
    for i, review in enumerate(reviews):
        request_count += 1
        
        if not isinstance(review, str) or not review.strip():
            results.append({
                'index': i,
                'error': 'Invalid or empty review'
            })
            continue
        
        try:
            result = analyzer.predict(review)
            result['index'] = i
            results.append(result)
        except Exception as e:
            results.append({
                'index': i,
                'error': str(e)
            })
    
    return jsonify({
        'total_reviews': len(reviews),
        'successful_predictions': len([r for r in results if 'error' not in r]),
        'results': results
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Development server
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"\n{'='*60}")
    print(f"🚀 Movie Sentiment Analyzer API")
    print(f"{'='*60}")
    print(f"Running on: http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"Model loaded: {analyzer is not None}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)