# Webcam Viewer with Face Detection — PRD

## Overview
A desktop application that displays a live webcam feed, detects human faces in real-time, and recognizes/identifies individual people across sessions.

## Tech Stack
- **Language:** Python 3
- **Library:** OpenCV (`opencv-python`) — open source, BSD license
- **Face Detection (Phase 2):** OpenCV DNN module with a pre-trained Caffe SSD model
- **Face Recognition (Phase 3):** OpenCV DNN face embedding model (`SFace`) for generating 128-d feature vectors per face

## Phases

### Phase 1 — Live Webcam Viewer ✅
- Open the default USB webcam (`/dev/video0`)
- Display the live feed in a resizable/maximizable GUI window
- Exit cleanly when the user presses `q` or closes the window
- Handle errors gracefully (camera not found, permission denied)

### Phase 2 — Face Detection ✅
- Load a face detection model at startup
- For each frame, detect faces and draw green bounding rectangles
- Display confidence scores on each detection
- Adjustable confidence threshold via slider (default 60%)
- Threshold value shown as overlay text on the video frame
- Labels clamped to stay within the visible frame
- Maintain real-time performance on low-resolution input

### Phase 3 — Face Recognition & Identification
- **Face embeddings:** For each detected face, compute a 128-dimensional feature vector (embedding) using OpenCV's SFace model. This is the standard approach for distinguishing between individuals — each person's face maps to a roughly consistent point in embedding space.
- **Persistent face database:** Store known face data in a local JSON file (`faces_db.json`). Each entry contains:
  - A unique ID
  - An auto-generated display name
  - A list of embedding vectors collected over time (multiple samples improve matching accuracy)
- **Cross-session persistence:** The database file is loaded at startup and updated during runtime. Different runs of the program contribute to the same database, so recognition improves over time.
- **Face matching:** On each frame, compare each detected face's embedding against all known faces using cosine similarity. If the similarity exceeds a recognition threshold, identify as that person. Otherwise, register as a new person.
- **Auto-generated names:** When a new face is first seen, assign a consistent, memorable name from a predefined list (e.g., "Atlas", "Nova", "Echo", "Sage"...). The name stays permanently tied to that face's database entry.
- **On-screen display:** Show the assigned name next to each face's bounding rectangle (alongside the existing confidence percentage).
- **Clear database:** Provide a keyboard shortcut (`c` key) to wipe the entire face database and start fresh. Confirm in the terminal output.
- **Model auto-download:** `run.sh` downloads the SFace model file automatically, same as the detection model.

## Requirements
- Ubuntu (tested on 24.04+)
- Python 3.10+
- USB webcam (any resolution)
- No GPU required

## How to Run
```bash
./run.sh
```
