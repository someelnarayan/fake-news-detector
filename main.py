import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

# -------------------------------
# Load Pre-trained Model
# -------------------------------

BASE_DIR = os.path.dirname(__file__)

vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))

# -------------------------------
# FastAPI Setup
# -------------------------------

app = FastAPI(
    title="Fake News Detector API",
    description="Detects whether news is REAL or FAKE",
    version="1.0.0"
)

class Article(BaseModel):
    text: str

# -------------------------------
# Prediction Endpoint
# -------------------------------

@app.post("/detect/")
def detect_fake_news(article: Article):
    
    vectorized_text = vectorizer.transform([article.text])
    
    prediction = model.predict(vectorized_text)[0]
    confidence = np.max(model.predict_proba(vectorized_text))

    return {
        "prediction": prediction,
        "confidence": float(confidence)
    }

# -------------------------------
# Health Check
# -------------------------------

@app.get("/")
def read_root():
    return {"message": "API is running 🚀"}