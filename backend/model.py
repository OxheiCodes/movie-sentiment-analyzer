import pickle
import re
from html import unescape
import time
import os

class SentimentAnalyzer:
    """Sentiment analysis model wrapper for production use"""
    
    def __init__(self, model_path='models'):
        """Initialize model and vectorizer"""
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load trained model and vectorizer"""
        try:
            model_file = os.path.join(self.model_path, 'lr_model.pkl')
            vectorizer_file = os.path.join(self.model_path, 'tfidf_vectorizer.pkl')
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("✅ Models loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove HTML entities
        text = unescape(text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def predict(self, review_text):
        """
        Predict sentiment for a review
        
        Args:
            review_text (str): Movie review text
            
        Returns:
            dict: Prediction results with sentiment, confidence, and timing
        """
        # Validate input
        if not review_text or not isinstance(review_text, str):
            return {
                'error': 'Invalid input: review text must be a non-empty string'
            }
        
        try:
            # Preprocess
            clean_text = self.preprocess_text(review_text)
            
            # Vectorize
            tfidf_features = self.vectorizer.transform([clean_text])
            
            # Predict
            start_time = time.time()
            prediction = self.model.predict(tfidf_features)[0]
            probability = self.model.predict_proba(tfidf_features)[0]
            prediction_time = (time.time() - start_time) * 1000
            
            # Format result
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            confidence = float(max(probability) * 100)
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 2),
                'prediction_time_ms': round(prediction_time, 2),
                'label': int(prediction)  # 0=Negative, 1=Positive
            }
        
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}'
            }

# Test the analyzer if run directly
if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    
    analyzer = SentimentAnalyzer()
    
    test_reviews = [
        "I loved this movie!",
        "Terrible waste of time",
        "It was okay, nothing special"
    ]
    
    for review in test_reviews:
        result = analyzer.predict(review)
        print(f"\nReview: '{review}'")
        print(f"Result: {result}")