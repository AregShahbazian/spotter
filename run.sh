#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

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

"$VENV_DIR/bin/python" webcam.py
