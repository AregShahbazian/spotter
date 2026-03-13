# Webcam Viewer with Face Detection — PRD

## Overview
A desktop application that displays a live webcam feed and (in Phase 2) detects and highlights human faces in real-time.

## Tech Stack
- **Language:** Python 3
- **Library:** OpenCV (`opencv-python`) — open source, BSD license
- **Face Detection (Phase 2):** OpenCV DNN module with a pre-trained Caffe model (or Haar cascades as fallback)

## Phases

### Phase 1 — Live Webcam Viewer
- Open the default USB webcam (`/dev/video0`)
- Display the live feed in a resizable GUI window
- Exit cleanly when the user presses `q` or closes the window
- Handle errors gracefully (camera not found, permission denied)

### Phase 2 — Face Detection (future)
- Load a face detection model at startup
- For each frame, detect faces and draw bounding rectangles
- Display confidence scores on each detection
- Maintain real-time performance on low-resolution input

## Requirements
- Ubuntu (tested on 24.04+)
- Python 3.10+
- USB webcam (any resolution)
- No GPU required

## How to Run
```bash
./run.sh
```
