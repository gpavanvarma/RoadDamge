import base64
import os
import cv2
import numpy as np
import json
import pickle
import threading
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-123'

# --- Database Config (SQLite local, PostgreSQL on Render) ---
db_url = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite')
# Fix for Render: SQLAlchemy 1.4+ requires "postgresql://" not "postgres://"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- Database Models ---
class User(UserMixin, db.Model):
    __tablename__ = 'app_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists')
        else:
            new_user = User(username=username, password=generate_password_hash(password, method='scrypt'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', name=current_user.username)

@app.route('/detect', methods=['POST'])
@login_required
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
@login_required
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
@login_required
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

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
