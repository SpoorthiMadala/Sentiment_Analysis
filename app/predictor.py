import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import clean_text

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, '..', 'data', 'reviews.csv')
    df = pd.read_csv(data_path)

    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")

    df['cleaned_text'] = df['review'].apply(clean_text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    model = LogisticRegression()
    model.fit(X, y)

    # Create model directory if it doesn't exist
    model_dir = os.path.join(BASE_DIR, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "sentiment_model.pkl")
    joblib.dump((model, vectorizer), model_path)

# Train the model on import (optional)
train_model()

# Load the model and vectorizer after training
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', "sentiment_model.pkl")
model, vectorizer = joblib.load(model_path)

def predict_sentiment(text):
    cleaned = clean_text(text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)
    return prediction[0]
