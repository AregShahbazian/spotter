# Spotter — PRD

## Overview
A desktop application that displays a live webcam feed, detects human faces in real-time, recognizes/identifies individual people across sessions, and provides a rich side-panel UI with sighting history, thumbnails, and toast notifications. Supports English and Russian languages.

## Tech Stack
- **Language:** Python 3
- **Library:** OpenCV (`opencv-python`) — open source, BSD license
- **Face Detection:** OpenCV DNN module with YuNet model
- **Face Recognition:** OpenCV DNN face embedding model (`SFace`) for generating 128-d feature vectors per face
- **Text Rendering:** Pillow (PIL) for Unicode/Cyrillic text support (optional, graceful fallback)
- **Packaging:** PyInstaller (single-folder bundle with all dependencies)

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

### Phase 3 — Face Recognition & Identification ✅
- **Face embeddings:** For each detected face, compute a 128-dimensional feature vector (embedding) using OpenCV's SFace model. Each person's face maps to a roughly consistent point in embedding space.
- **Persistent face database:** Store known face data in a local JSON file (`faces_db.json`). Each entry contains:
  - An auto-generated display name
  - A list of embedding vectors collected over time (up to 20 per person)
  - A list of sighting records (start time, end time, duration)
- **Cross-session persistence:** The database file is loaded at startup and updated during runtime. Recognition improves over time as more embeddings are collected.
- **Face matching:** On each frame, compare each detected face's embedding against all known faces using cosine similarity. If the similarity exceeds the recognition threshold, identify as that person. Otherwise, begin a pending track.
- **IoU-based frame-to-frame tracking:** Faces are tracked across frames using bounding box Intersection over Union (IoU), avoiding redundant re-identification every frame. Tracks maintain state (bounding box, embedding, frames seen, missed count).
- **Pending registration gate:** New faces must be observed for 20 consecutive frames (~0.7s at 30fps) before being registered. This prevents transient false detections from polluting the database.
- **Auto-generated names:** When a new face is confirmed, assign a memorable name from a predefined list of 30 names (Atlas, Nova, Echo, Sage, etc.). Names cycle with suffixes (e.g., "Atlas 2") if all base names are used.
- **On-screen display:** Show the assigned name (or "?" for pending) and confidence next to each face's bounding box. Pending faces use yellow boxes, confirmed faces use green.
- **Sighting tracking:** Record when each person enters and exits the frame. Sightings with duration < 0.5s are discarded. Sightings are persisted in the JSON database.
- **Side panel UI:** A 300px dark panel on the right side of the video showing:
  - "People" header
  - Each known person with expand/collapse triangle, name, and status
  - "LIVE (N)" indicator for currently visible people with sighting count
  - Expandable sighting history per person (most recent 15 entries)
  - Current live session duration shown as "NOW" with elapsed time
  - Past sighting timestamps and durations
  - Scrollable with mouse wheel, clickable to expand/collapse
- **Clear database:** `c` key wipes the entire face database, all tracks, and starts fresh.
- **Model auto-download:** `run.sh` downloads the SFace and YuNet model files automatically.

### Phase 4 — Identity Hardening ✅
Addresses identity confusion and flickering by adding multiple layers of verification:

- **Stricter cosine threshold:** Raised from 0.363 to 0.40 to reduce false positive matches.
- **Ambiguity rejection:** When the best and second-best DB match scores are within 0.05 (MATCH_MARGIN), the match is rejected as ambiguous.
- **Embedding quality gate:** Before adding a new embedding to the DB, check its cosine similarity against the existing average. Reject if below 0.25 (TRACK_VERIFY_THRESHOLD).
- **Minimum face size filter:** Detections smaller than 60px in width or height are skipped entirely.
- **Embedding verification after IoU match:** When a confirmed track is spatially matched via IoU, verify that the new face embedding matches the tracked identity. If similarity < 0.25, reject the match.
- **Embedding-based track recovery:** When a confirmed track loses IoU (e.g., head turn), attempt to re-link it to unmatched detections using embedding similarity (threshold 0.35).
- **Outlier-filtered averaging on promotion:** Discard outlier embeddings before averaging when promoting pending tracks.
- **Longer observation period:** Pending frames increased from 10 to 20.

### Phase 5 — Thumbnails, Notifications & Localization ✅

- **Face thumbnails:** 60x60px face crops saved to `thumbs/` as JPEGs on registration and re-entry. Displayed in the side panel when expanded.
- **Toast notifications:** On-screen overlay messages — "Welcome, {name}!" (3s), "{name} arrived" (2s), "{name} left" (2s). Centered at bottom with black outline.
- **Language toggle (EN/RU):** Press `l` to cycle. All on-screen UI text translated. Cyrillic rendered via Pillow. Current language shown in HUD.

### Phase 6 — Ubuntu Packaging 🔲
Package Spotter as a standalone `.deb` installer for Ubuntu 22.04+.

- **Build tool:** PyInstaller bundles Python interpreter, OpenCV, Pillow, numpy, and all code into a self-contained folder. No system Python or pip needed on the target machine.
- **Bundled assets:** ONNX model files (YuNet, SFace) are included inside the package. The user does not need to download anything separately.
- **`.deb` package:** The PyInstaller output is wrapped in a Debian package (`spotter_1.0.0_amd64.deb`). Double-click or `sudo dpkg -i` to install.
- **Install location:** `/opt/spotter/` (application files), `/usr/local/bin/spotter` (launcher symlink).
- **Desktop integration:** A `.desktop` file is installed to `/usr/share/applications/`. Spotter appears in the Ubuntu Activities search, application grid, and can be pinned to the dock/favorites. Includes an app icon.
- **Data directory:** On first run, `faces_db.json` and `thumbs/` are created in `~/.local/share/spotter/` (XDG-compliant), not next to the binary.
- **Uninstall:** `sudo dpkg -r spotter` removes everything cleanly.
- **Build script:** `build-ubuntu.sh` automates the full build → package pipeline on the developer's Ubuntu machine.

### Phase 7 — Windows Packaging 🔲
Package Spotter as a `.exe` installer for Windows 10/11.

- **Build tool:** PyInstaller on a Windows environment (GitHub Actions CI with `windows-latest` runner).
- **Installer:** Inno Setup wraps the PyInstaller output into a familiar `SpotterSetup.exe` install wizard (Next → Next → Install → Finish).
- **Desktop integration:** Creates a Start Menu entry, optional desktop shortcut, and an uninstaller in Add/Remove Programs.
- **Data directory:** `%APPDATA%\Spotter\` for `faces_db.json` and `thumbs/`.
- **Code signing:** Unsigned initially (SmartScreen warning on first run). Code signing certificate can be added later to remove the warning.
- **CI/CD:** GitHub Actions workflow builds both Ubuntu and Windows packages on each tagged release. Artifacts uploaded to GitHub Releases.

## Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `c` | Clear face database, tracks, thumbnails, and start fresh |
| `l` | Toggle language between English and Russian |

## File Structure
| File/Dir | Purpose |
|----------|---------|
| `webcam.py` | Main application |
| `run.sh` | Dev setup script: creates venv, installs deps, downloads models, runs app |
| `build-ubuntu.sh` | Build script: creates `.deb` package for Ubuntu |
| `spotter.spec` | PyInstaller spec file |
| `spotter.desktop` | Linux desktop entry file |
| `assets/spotter.svg` | Application icon |
| `model/` | YuNet and SFace ONNX model files |
| `faces_db.json` | Persistent face database (dev mode, in working dir) |
| `thumbs/` | Face thumbnail images (dev mode, in working dir) |
| `PRD.md` | This document |

## Requirements
- Ubuntu 22.04+ or Windows 10/11
- USB webcam (any resolution)
- No GPU required
- No Python installation required (bundled in package)

## How to Run (Development)
```bash
./run.sh
```

## How to Install (Ubuntu)
```bash
./build-ubuntu.sh
sudo dpkg -i dist/spotter_1.0.0_amd64.deb
```
Then search "Spotter" in Activities or run `spotter` from terminal.
