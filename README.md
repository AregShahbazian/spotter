# Spotter

Live face detection and recognition from a webcam, with a persistent memory of
who it has seen. Point it at a room and it draws boxes around faces, gives
each new person an auto-generated name (Atlas, Nova, Echo, …), recognises them
again on later runs, and keeps a sighting log with thumbnails in a side panel.
Everything runs locally on the CPU — no cloud, no GPU, no accounts.

## How it works

1. **Detection** — each frame goes through OpenCV's YuNet DNN detector;
   detections under the confidence slider (default 60%) or smaller than 60 px
   are dropped.
2. **Embedding** — every kept face is aligned and turned into a 128-d SFace
   feature vector. Identity is a cosine-similarity match against the stored
   people (threshold 0.40); a match whose runner-up is within 0.05 is rejected
   as ambiguous.
3. **Tracking** — faces are followed frame to frame by bounding-box IoU so they
   are not re-identified every frame. An IoU match is sanity-checked against the
   track's embedding, and a track that loses IoU (head turn) can be re-linked by
   embedding alone.
4. **Registration gate** — an unknown face must be seen for 20 consecutive
   frames (~0.7 s) before it is registered; the stored embedding is an
   outlier-filtered average of those frames. Known people accumulate up to 20
   embeddings over time, each quality-gated, so recognition improves with use.

Sightings (enter/leave times) are recorded per person, toasts announce
arrivals and departures, and the UI is available in English and Russian.

Keys: `q` quit · `c` clear the database and thumbnails · `l` toggle EN/RU.

## Run from source

```bash
./run.sh
```

Creates a `.venv`, installs `opencv-python` and `Pillow`, downloads the two
ONNX models from the OpenCV model zoo into `model/`, and starts the app.
Needs Python 3 and a webcam. In dev mode `faces_db.json` and `thumbs/` are written next to
`webcam.py`.

## Build the binary

```bash
./build-ubuntu.sh            # PyInstaller bundle → dist/spotter_1.0.0_amd64.deb
sudo dpkg -i dist/spotter_1.0.0_amd64.deb
```

The `.deb` installs to `/opt/spotter`, adds a launcher entry, and stores data
under `~/.local/share/spotter/`. Pushing a `v*` tag runs the GitHub Actions
workflow, which builds the Ubuntu `.deb` and a Windows Inno Setup installer
(`SpotterSetup.exe`, data in `%APPDATA%\Spotter\`) and attaches both to a
GitHub Release.

## Privacy

Spotter never sends anything anywhere. Recognition data lives in
`faces_db.json` (SFace embeddings plus names and sighting times) and face
crops in `thumbs/`, both local files. Both are gitignored, and `c` wipes them.

## Docs

The phase-by-phase design record is in [docs/PRD.md](docs/PRD.md).

## License

MIT — see [LICENSE](LICENSE).
