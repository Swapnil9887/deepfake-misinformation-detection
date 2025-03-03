import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import cv2
import matplotlib.pyplot as plt
from app import DeepFakeDetector  # Ensure this matches your model file

# Load Model
MODEL_PATH = "models/deepfake_model.pth"
model = DeepFakeDetector()  # Initialize model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()  # Set to evaluation mode

# Function to randomly select an image from subdirectories
def get_random_image():
    dataset_folders = ["data/real/real_images", "data/deepfake/deepfake_images"]  # Update paths
    selected_folder = random.choice(dataset_folders)  # Pick either real or deepfake
    image_files = []

    # Search for images inside the selected folder
    for root, _, files in os.walk(selected_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check valid image types
                image_files.append(os.path.join(root, file))

    if not image_files:
        raise Exception(f"❌ No images found in {selected_folder}. Ensure images exist!")

    selected_image = random.choice(image_files)
    return selected_image  # Return full image path

# Get a random image from the dataset
IMAGE_PATH = get_random_image()
print(f"🖼️ Selected Random Image: {IMAGE_PATH}")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load Image & Convert
image = Image.open(IMAGE_PATH)
input_tensor = transform(image).unsqueeze(0)

# Make Prediction
output = model(input_tensor)
prediction = torch.argmax(output, dim=1).item()
result_text = "Deepfake" if prediction == 1 else "Real"

# Load Accuracy (Assumes accuracy was saved)
accuracy_file = "models/accuracy.txt"
if os.path.exists(accuracy_file):
    with open(accuracy_file, "r") as f:
        model_accuracy = f.read().strip()
else:
    model_accuracy = "Accuracy not recorded"

# Print Result with Accuracy
print(f"✅ Prediction: {result_text} (from {IMAGE_PATH})")
print(f"📊 Model Accuracy: {model_accuracy}")

# Show Image Using Matplotlib
plt.imshow(Image.open(IMAGE_PATH))
plt.title(f"Prediction: {result_text} (Accuracy: {model_accuracy})")
plt.axis("off")
plt.show()

# Show Image Using OpenCV with Accuracy Overlay
image_cv = cv2.imread(IMAGE_PATH)
cv2.putText(image_cv, f"Prediction: {result_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(image_cv, f"Accuracy: {model_accuracy}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow("DeepFake Detection Result", image_cv)
cv2.waitKey(0)  # Press any key to close window
cv2.destroyAllWindows()




