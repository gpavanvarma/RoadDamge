import base64
import os
import cv2
import numpy as np
import json
import pickle
import threading
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-123'

# --- YOLO Model Setup ---
MODEL_PATH = "model/best.pt"
model = None
model_lock = threading.Lock()

try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading YOLOv8 model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- Routes ---

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('index.html', name='Guest')

@app.route('/detect', methods=['POST'])
def detect():
    if not model:
        return jsonify({'error': 'Model not loaded or found'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Optional: Resize large images to avoid memory crashes
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        with model_lock:
            results = model(img)
            annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        detections = []
        # Mapping for RDD2022 codes to readable names
        class_map = {
            'D00': 'Longitudinal Crack',
            'D10': 'Transverse Crack',
            'D20': 'Alligator Crack',
            'D40': 'Pothole',
            'Block crack': 'Block Crack',
            'Repair': 'Repaired Road'
        }

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_label = results[0].names[class_id]
            # Use mapped name if available, otherwise raw label
            label = class_map.get(raw_label, raw_label)
            detections.append({'label': label, 'confidence': f"{conf:.2f}"})

        return jsonify({'image': img_base64, 'detections': detections})

    except Exception as e:
        print(f"CRITICAL ERROR in /detect: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Analytics API ---

@app.route('/api/metrics')
def metrics():
    try:
        with open('model/metrics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"ERROR in /api/metrics: {e}")
        return jsonify({'error': str(e)}), 500

def make_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj

@app.route('/api/history')
def history():
    history_data = {}
    files = {
        'v5': 'model/v5_history.pckl',
        'v7': 'model/v7_history.pckl',
        'v8': 'model/v8_history.pckl'
    }
    for key, filepath in files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                clean_data = make_serializable({
                    'accuracy': data.get('accuracy', []),
                    'loss': data.get('loss', [])
                })
                history_data[key] = clean_data
            except Exception as e:
                print(f"Error loading history {filepath}: {e}")
                history_data[key] = None
    return jsonify(history_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
