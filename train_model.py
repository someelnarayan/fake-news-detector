import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading datasets...")

# Load datasets
fake_df = pd.read_csv("Fake.csv", encoding="latin1")
true_df = pd.read_csv("True.csv", encoding="latin1")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine
df = pd.concat([fake_df, true_df])

# Keep only required columns
df = df[["text", "label"]]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

print("Vectorizing...")

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

print("Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Saving model...")

joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(model, "logistic_regression_model.joblib")

print("✅ DONE — model files created!")