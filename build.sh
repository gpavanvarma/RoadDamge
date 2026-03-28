#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

# Ultralytics automatically installs the 'opencv-python' package which requires 
# system GUI libraries (like libGL) that aren't present on cloud environments.
# We uninstall it and ensure only 'opencv-python-headless' remains.
echo "Replacing opencv-python with opencv-python-headless..."
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless

echo "Build complete."
