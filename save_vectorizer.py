import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load your training dataset (Replace with actual training data)
news_train_texts = ["This is a sample news article", "Fake news spreads misinformation", "AI is transforming journalism"]

# ✅ Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(news_train_texts)  # Fit with your training texts

# ✅ Save vectorizer as pickle file
vectorizer_path = "models/vectorizer.pkl"
with open(vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

print(f"✅ TF-IDF Vectorizer saved at: {vectorizer_path}")
