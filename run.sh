#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
MODEL_DIR="model"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Install it with: sudo apt install python3"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c "import cv2" 2>/dev/null; then
    echo "Installing opencv-python..."
    "$VENV_DIR/bin/pip" install opencv-python
fi

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/face_detection_yunet_2023mar.onnx" ]; then
    echo "Downloading YuNet face detection model..."
    curl -L -o "$MODEL_DIR/face_detection_yunet_2023mar.onnx" "$YUNET_URL"
fi

if [ ! -f "$MODEL_DIR/face_recognition_sface_2021dec.onnx" ]; then
    echo "Downloading SFace face recognition model..."
    curl -L -o "$MODEL_DIR/face_recognition_sface_2021dec.onnx" "$SFACE_URL"
fi

echo "Models ready."

"$VENV_DIR/bin/python" webcam.py
