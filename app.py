import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Ensure 'uploads' directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load DeepFake Detection Model (Images)
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ✅ Dynamically calculate FC layer size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.pool(self.relu(self.conv1(dummy_input)))
            self.flatten_size = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# ✅ Load the trained deepfake model
MODEL_PATH = "models/deepfake_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Deepfake Model file not found at '{MODEL_PATH}'")

device = torch.device("cpu")
deepfake_model = DeepFakeDetector().to(device)
deepfake_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
deepfake_model.eval()

# ✅ Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ✅ Load Fake News Detection Model & Vectorizer
NEWS_MODEL_PATH = "models/fake_news_model.pkl"

if not os.path.exists(NEWS_MODEL_PATH):
    raise FileNotFoundError(f"❌ Fake News Model file not found at '{NEWS_MODEL_PATH}'")

# ✅ Load news detection model (Pipeline with TF-IDF + Classifier)
with open(NEWS_MODEL_PATH, "rb") as model_file:
    news_pipeline = pickle.load(model_file)

# ✅ Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Image Upload Route (DeepFake Detection)
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ✅ Process Image
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # ✅ Make Prediction
    output = deepfake_model(image)
    prediction = torch.argmax(output, dim=1).item()
    result = "REAL ✅" if prediction == 1 else "FAKE ❌"

    return render_template("result.html", prediction=result, image_path=f"uploads/{filename}")

# ✅ FIXED: News Text Prediction Route (Fake News Detection)
@app.route("/predict_news", methods=["POST"])
def predict_news():
    news_text = request.form.get("news_text")
    if not news_text:
        return jsonify({"error": "No text entered!"}), 400

    try:
        # ✅ Ensure text is in a **list**, as `predict()` requires an array-like input
        processed_text = [news_text.strip()]

        # ✅ Predict using trained pipeline
        prediction = news_pipeline.predict(processed_text)[0]

        # ✅ FIXED Label Mapping: Ensure 0 = Fake, 1 = Real
        result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"

        return render_template("result.html", news_text=news_text, news_prediction=result)

    except Exception as e:
        return jsonify({"error": f"News prediction failed: {str(e)}"}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
