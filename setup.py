import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

def initialize_model():
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'model.joblib')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    print("Initializing sentiment analysis model...")
    
    reviews = [
        "The food was absolutely amazing and delicious!",
        "Great service, wonderful atmosphere, highly recommend!",
        "Best restaurant I've ever been to, fantastic experience!",
        "Delicious food and friendly staff, will come again!",
        "Outstanding quality, fresh ingredients, loved it!",
        "Excellent menu, tasty dishes, great value for money!",
        "Perfect dining experience, superb food and service!",
        "The ambiance was lovely and the food was exceptional!",
        "Highly satisfied with the quality and taste!",
        "Amazing flavors, generous portions, worth every penny!",
        "Terrible food, very disappointing experience.",
        "Worst restaurant ever, would not recommend at all.",
        "The service was awful and the food was cold.",
        "Horrible experience, never going back again!",
        "Disgusting food, poor hygiene, avoid this place!",
        "Overpriced and tasteless, complete waste of money.",
        "Rude staff, terrible service, very unsatisfied.",
        "Food was undercooked and service was slow.",
        "Unpleasant atmosphere, bad food quality.",
        "Disappointing meal, not worth the price at all.",
        "The food was okay, nothing special.",
        "Average restaurant, neither good nor bad.",
        "It was fine, met basic expectations.",
        "Not bad but not great either, just mediocre.",
        "Decent food but could be better.",
        "Acceptable quality, nothing to complain about.",
        "Standard restaurant experience, nothing remarkable.",
        "The meal was alright, nothing extraordinary.",
        "Fair enough, moderate quality and service.",
        "So-so experience, neither impressed nor disappointed."
    ]
    
    labels = [
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative",
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral"
    ]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training model on demo dataset...")
    pipeline.fit(reviews, labels)
    
    joblib.dump(pipeline, model_path)
    print(f"Model initialized and saved to {model_path}")
    print("Setup complete! You can now run 'python app.py' to start the application.")

if __name__ == "__main__":
    initialize_model()
