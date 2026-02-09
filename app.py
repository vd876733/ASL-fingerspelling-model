from flask import Flask, request, jsonify, send_from_directory
import os
import io
import base64
from PIL import Image
import numpy as np
import cv2
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='/static')

# Initialize model objects (paths relative to this file)
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'keras_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'Model', 'labels.txt')

detector = HandDetector(maxHands=2)
classifier = None
labels = []

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        classifier = Classifier(MODEL_PATH, LABELS_PATH)
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print('Model load error:', e)
else:
    print('Model files not found. /predict will return mock responses until model is provided.')

offset = 20
imgSize = 300


def read_image_from_base64(data_url):
    # data_url expected like 'data:image/png;base64,...'
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_and_predict(bgr_image):
    # Use hand detector to crop to hand area, then mimic logic from PYTHON.py
    hands, _ = detector.findHands(bgr_image, draw=False)
    if not hands:
        return None

    # use first hand
    hand = hands[0]
    x, y, w, h = hand['bbox']
    y1 = max(0, y - offset)
    y2 = min(bgr_image.shape[0], y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(bgr_image.shape[1], x + w + offset)
    imgCrop = bgr_image[y1:y2, x1:x2]
    if imgCrop.size == 0:
        return None

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    aspectRatio = h / w if w != 0 else 1

    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / w if w != 0 else 1
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    if classifier is None:
        # mock prediction
        return {'label': 'A', 'confidence': 0.85}

    try:
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        # classifier returns array of confidences, index is argmax
        conf = float(prediction[index]) if isinstance(prediction, (list, tuple, np.ndarray)) else float(max(prediction))
        label = labels[index] if index < len(labels) else str(index)
        return {'label': label, 'confidence': conf}
    except Exception as e:
        print('Prediction error:', e)
        return None


@app.route('/')
def index():
    # serve the main HTML file
    return send_from_directory(BASE_DIR, 'aiml.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        bgr = read_image_from_base64(data['image'])
    except Exception as e:
        return jsonify({'error': 'Invalid image data', 'detail': str(e)}), 400

    result = preprocess_and_predict(bgr)
    if not result:
        return jsonify({'error': 'No hand detected or prediction failed'}), 200

    return jsonify({'letter': result['label'], 'confidence': round(result['confidence'] * 100, 2)})


if __name__ == '__main__':
    # Run on all interfaces so frontend served from same machine can access
    app.run(host='0.0.0.0', port=5000, debug=False)
