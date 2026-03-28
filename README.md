# Road Damage Detection Web App

A modern web application for detecting road damage (potholes, cracks, etc.) using YOLOv8 and Flask.

## Features
- **Modern UI**: Clean, responsive interface with drag-and-drop support.
- **Fast Detection**: Uses YOLOv8 for accurate, real-time object detection.
- **Render Ready**: Configured for easy deployment to Render (Free Tier compatible).

## Tech Stack
- **Backend**: Python, Flask, OpenCV, Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Gunicorn (for production)

## Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start at `http://localhost:5000`.

3. **Usage**:
   - Open your browser and go to `http://localhost:5000`.
   - Upload an image (drag & drop or click to browse).
   - View the detected damage and download the result.

## Deploying to Render

1. Push this code to a GitHub repository.
2. Log in to [Render](https://render.com).
3. Create a new **Web Service**.
4. Connect your GitHub repository.
5. Render will auto-detect the configuration:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click **Deploy Web Service**.

> **Note**: This app is optimized for Render's Free Tier (512MB RAM). The YOLOv8 model is loaded globally to save memory per request.
