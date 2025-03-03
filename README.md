# 🔍 Deepfake & Fake News Detection  

This project detects **deepfake images/videos** and **fake news articles** using AI models. It leverages deep learning and NLP techniques to identify manipulated media and misinformation.  

## 📌 Features  
- 🖼️ **Deepfake Detection**: Identify real vs. fake images/videos using a trained CNN model.  
- 📰 **Fake News Detection**: Classify news articles as real or fake using NLP-based machine learning models.  
- 🌐 **Web Interface**: Interactive Flask-based UI for users to test images, videos, and news articles.  
- 📊 **Accuracy Reports**: Track model performance and detection confidence.  

---

## ⚙️ Installation  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/Deepfake-Detection.git
cd Deepfake-Detection
2️⃣ Set Up Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 Usage
🎭 Deepfake Image/Video Detection
bash
Copy
Edit
python test_model.py --image path/to/image.jpg
python test_model.py --video path/to/video.mp4
📰 Fake News Detection
bash
Copy
Edit
python test_fake_news.py --text "News article content here"
🌐 Run the Web App
bash
Copy
Edit
python app.py
Then open http://127.0.0.1:5000/ in your browser.
