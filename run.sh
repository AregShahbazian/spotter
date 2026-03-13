#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
MODEL_DIR="model"
PROTO_URL="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL="https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

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

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

if [ ! -f "$MODEL_DIR/deploy.prototxt" ]; then
    echo "Downloading face detection model..."
    curl -L -o "$MODEL_DIR/deploy.prototxt" "$PROTO_URL"
    curl -L -o "$MODEL_DIR/res10_300x300_ssd_iter_140000.caffemodel" "$MODEL_URL"
    echo "Model downloaded."
fi

"$VENV_DIR/bin/python" webcam.py
