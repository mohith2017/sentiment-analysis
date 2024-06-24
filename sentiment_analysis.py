import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from preprocess_data import preprocess_text


df = pd.read_csv('data/Processed Data.csv')

X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vectorized, y_train)

y_pred = rf_model.predict(X_test_vectorized)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Predict sentiment function
def predict_sentiment(review):
    processed = preprocess_text(review)
    vectorized = vectorizer.transform([processed])
    prediction = rf_model.predict(vectorized)
    return prediction[0]

new_review = "This movie was absolutely fantastic! I loved every minute of it."
print(f"\nSentiment of '{new_review}': {predict_sentiment(new_review)}")