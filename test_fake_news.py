import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Load the dataset
DATASET_PATH = "news_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# ✅ Ensure dataset has 'text' and 'label' columns
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("❌ Dataset must contain 'text' and 'label' columns!")

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# ✅ Create TF-IDF + Logistic Regression pipeline
news_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression())
])

# ✅ Train the model
news_pipeline.fit(X_train, y_train)

# ✅ Test accuracy
y_pred = news_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

# ✅ Save the trained model
MODEL_PATH = "models/fake_news_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(news_pipeline, f)

print(f"✅ Model saved to {MODEL_PATH}")
