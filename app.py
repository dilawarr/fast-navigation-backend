from flask import Flask, request, jsonify
from torchvision import transforms, models
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import cv2
import os
import io

app = Flask(__name__)
CORS(app)

# Setup device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (modify as per your model)
class_names = ['A', 'B', 'C', 'D', 'F', 'Library']

# Define the model transformation pipeline
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the pre-trained model (resnet34 in your case)
model = models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Adjust output layer size to match your class_names length
model.load_state_dict(torch.load("models/landmark_model.pth"))
model = model.to(device)
model.eval()

# Preprocess function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image

def detect_bbox(img):
    """Detects the bounding box of the main object in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return x, y, w, h
    else:
        return None  # Or raise an exception, handle as appropriate


focal_px = 10080.0  # Calibrated focal length (from the notebook)
real_heights = { # From the notebook
    # 'A_Block': 2.0,
    # 'B_Block': 3.5,
    # 'C_Block': 4.0,
    # 'D_Block': 3.0,
    # 'F_Block': 2.5,
    'A': 2.0,
    'B': 3.5,
    'C': 4.0,
    'D': 3.0,
    'F': 2.5,
    'Library': 5.0
}

def estimate_distance(image_path, block_name):
    """Estimates the distance to an object in an image."""

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to open image")

    bbox = detect_bbox(img)
    if bbox is None:
        raise ValueError("Object not detected")
    x, y, w, h = bbox

    if block_name not in real_heights:
        raise ValueError(f"Unknown block: {block_name}")

    real_height = real_heights[block_name]
    distance = (real_height * focal_px) / h
    return distance, x, y, w, h, img.shape[:2]  # Return distance and bbox info

@app.route('/predict-landmark', methods=['POST'])
def predict_landmark():
    for i in request.files:
        print(i)
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    print("ded")
    file = request.files['image']
    image_bytes = file.read()
    preprocessed_image = preprocess_image(image_bytes)
    
    # Predict
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs, 1)
    
    predicted_landmark = class_names[predicted.item()]
    
    return jsonify({'predicted_landmark': predicted_landmark})

@app.route('/predict-distance', methods=['POST'])
def predict_distance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'block_name' not in request.form:
        return jsonify({'error': 'No block_name provided'}), 400

    image_file = request.files['image']
    block_name = request.form['block_name']

    # Save the image temporarily
    image_path = "temp_image.jpg"
    image_file.save(image_path)

    try:
        distance, x, y, w, h, (h0, w0) = estimate_distance(image_path, block_name)
        return jsonify({
            'distance': distance,
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
            'image_dimensions': {'height': h0, 'width': w0}
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500
    finally:
        # Clean up the temporary image
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route('/')
def index():
    return "Fast Navigation Backend"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
