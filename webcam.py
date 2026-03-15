"""Spotter — Live face detection, recognition, and tracking."""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# When frozen (PyInstaller), assets are in the bundle; data goes to ~/.local/share/spotter
if getattr(sys, "frozen", False):
    _BUNDLE_DIR = sys._MEIPASS
    if sys.platform == "win32":
        _DATA_DIR = os.path.join(os.environ.get("APPDATA",
                                 os.path.expanduser("~")), "Spotter")
    else:
        _DATA_DIR = os.path.join(os.environ.get("XDG_DATA_HOME",
                                 os.path.expanduser("~/.local/share")), "spotter")
    os.makedirs(_DATA_DIR, exist_ok=True)
else:
    _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = _BUNDLE_DIR

MODEL_DIR = os.path.join(_BUNDLE_DIR, "model")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
FACES_DB_FILE = os.path.join(_DATA_DIR, "faces_db.json")
DEFAULT_THRESHOLD = 60
COSINE_THRESHOLD = 0.40   # raised from 0.363 to reduce false matches
WINDOW_NAME = "Spotter"
MAX_EMBEDDINGS = 20
EMBEDDING_INTERVAL = 2.0  # seconds between storing embeddings for a known face
PANEL_WIDTH = 300
PANEL_BG = (30, 30, 30)
PANEL_HEADER_H = 35
ROW_HEIGHT = 28
SIGHTING_ROW_H = 20
THUMB_DIR = os.path.join(_DATA_DIR, "thumbs")
THUMB_SIZE = 60
TOAST_WELCOME_SEC = 3.0
TOAST_EVENT_SEC = 2.0

NAMES = [
    "Atlas", "Nova", "Echo", "Sage", "Orion",
    "Luna", "Blaze", "Coral", "Drift", "Ember",
    "Frost", "Haze", "Iris", "Jade", "Kai",
    "Lyra", "Mars", "Nyx", "Onyx", "Pearl",
    "Quinn", "Rune", "Sol", "Thorn", "Uma",
    "Vale", "Wren", "Xena", "Yara", "Zephyr",
]

I18N = {
    "en": {
        "hud": "Threshold: {t}%  |  Known: {n}  |  EN",
        "panel_header": "People",
        "live": "LIVE ({c})",
        "count": "({c})",
        "now": "  NOW  {d}",
        "welcome": "Welcome, {name}!",
        "arrived": "{name} arrived",
        "left": "{name} left",
        "db_cleared": "Face database cleared.",
        "new_face": "New face registered: {name}",
    },
    "ru": {
        "hud": "Порог: {t}%  |  Известно: {n}  |  RU",
        "panel_header": "Люди",
        "live": "ОНЛАЙН ({c})",
        "count": "({c})",
        "now": "  СЕЙЧАС  {d}",
        "welcome": "Привет, {name}!",
        "arrived": "{name} пришёл",
        "left": "{name} ушёл",
        "db_cleared": "База лиц очищена.",
        "new_face": "Новое лицо: {name}",
    },
}

_FONT_CACHE = {}


def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in font_paths:
        if os.path.isfile(p):
            font = ImageFont.truetype(p, size)
            _FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_text(img, text, org, font_scale, color_bgr, thickness=1):
    """Draw text with Unicode support. Falls back to cv2.putText for ASCII."""
    if not _PIL_AVAILABLE or all(ord(c) <= 127 for c in text):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color_bgr, thickness)
        return
    pil_size = max(10, int(font_scale * 30))
    font = _get_font(pil_size)
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left + 4, bottom - top + 4
    txt_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((-left + 2, -top + 2), text, font=font, fill=color_rgb)
    txt_np = np.array(txt_img)
    x, y = org[0], org[1] - (bottom - top)
    ih, iw = img.shape[:2]
    sx, sy = max(0, -x), max(0, -y)
    dx, dy = max(0, x), max(0, y)
    cw = min(tw - sx, iw - dx)
    ch = min(th - sy, ih - dy)
    if cw <= 0 or ch <= 0:
        return
    alpha = txt_np[sy:sy+ch, sx:sx+cw, 3:4].astype(np.float32) / 255.0
    fg = txt_np[sy:sy+ch, sx:sx+cw, :3][:, :, ::-1].astype(np.float32)
    bg = img[dy:dy+ch, dx:dx+cw].astype(np.float32)
    img[dy:dy+ch, dx:dx+cw] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)


def text_size(text, font_scale, thickness=1):
    """Get (width, height) of text, supporting Unicode."""
    if not _PIL_AVAILABLE or all(ord(c) <= 127 for c in text):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, thickness)
        return tw, th
    pil_size = max(10, int(font_scale * 30))
    font = _get_font(pil_size)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def tr(lang, key, **kwargs):
    return I18N[lang][key].format(**kwargs)


class FaceDB:
    """Persistent face database stored as JSON."""

    def __init__(self, path):
        self.path = path
        self.faces = []
        self.next_name_idx = 0
        self._dirty = False
        self._avg_cache = {}
        self._last_embed_time = {}
        self._thumb_cache = {}
        self._load()
        self._load_thumbs()

    def _load(self):
        if os.path.isfile(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.faces = data.get("faces", [])
            self.next_name_idx = data.get("next_name_idx", 0)
            for face in self.faces:
                if "sightings" not in face:
                    face["sightings"] = []
        self._rebuild_cache()

    def _rebuild_cache(self):
        self._avg_cache.clear()
        for i, face in enumerate(self.faces):
            if face["embeddings"]:
                avg = np.array(face["embeddings"], dtype=np.float32).mean(axis=0).reshape(1, -1)
                self._avg_cache[i] = avg

    def _load_thumbs(self):
        if not os.path.isdir(THUMB_DIR):
            return
        for face in self.faces:
            name = face["name"]
            path = os.path.join(THUMB_DIR, f"{name}.jpg")
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    self._thumb_cache[name] = img

    def save_thumb(self, name, frame, bbox):
        os.makedirs(THUMB_DIR, exist_ok=True)
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
        cv2.imwrite(os.path.join(THUMB_DIR, f"{name}.jpg"), thumb)
        self._thumb_cache[name] = thumb

    def get_thumb(self, name):
        return self._thumb_cache.get(name)

    def save(self):
        if not self._dirty:
            return
        with open(self.path, "w") as f:
            json.dump({"faces": self.faces, "next_name_idx": self.next_name_idx}, f)
        self._dirty = False

    def save_force(self):
        with open(self.path, "w") as f:
            json.dump({"faces": self.faces, "next_name_idx": self.next_name_idx}, f)
        self._dirty = False

    def clear(self):
        self.faces.clear()
        self.next_name_idx = 0
        self._avg_cache.clear()
        self._last_embed_time.clear()
        self._thumb_cache.clear()
        if os.path.isdir(THUMB_DIR):
            for f in os.listdir(THUMB_DIR):
                os.remove(os.path.join(THUMB_DIR, f))
        self.save_force()

    def find_match(self, recognizer, embedding):
        scores = []
        for i, avg in self._avg_cache.items():
            score = recognizer.match(embedding, avg, cv2.FaceRecognizerSF_FR_COSINE)
            scores.append((score, i))
        if not scores:
            return -1, -1
        scores.sort(reverse=True)
        best_score, best_idx = scores[0]
        if best_score < COSINE_THRESHOLD:
            return -1, best_score
        # Ambiguity rejection: if second-best is too close, refuse match
        if len(scores) > 1:
            second_score = scores[1][0]
            if best_score - second_score < MATCH_MARGIN:
                return -1, best_score
        return best_idx, best_score

    def add_embedding(self, face_idx, embedding, recognizer=None):
        now = time.monotonic()
        last = self._last_embed_time.get(face_idx, 0)
        if now - last < EMBEDDING_INTERVAL:
            return
        # Quality gate: reject embeddings too dissimilar from existing average
        if recognizer is not None and face_idx in self._avg_cache:
            emb_flat = embedding.reshape(1, -1)
            sim = recognizer.match(emb_flat, self._avg_cache[face_idx],
                                   cv2.FaceRecognizerSF_FR_COSINE)
            if sim < TRACK_VERIFY_THRESHOLD:
                return
        self._last_embed_time[face_idx] = now

        emb_list = self.faces[face_idx]["embeddings"]
        if len(emb_list) >= MAX_EMBEDDINGS:
            emb_list.pop(0)
        emb_list.append(embedding.flatten().tolist())

        avg = np.array(emb_list, dtype=np.float32).mean(axis=0).reshape(1, -1)
        self._avg_cache[face_idx] = avg
        self._dirty = True

    def register(self, embedding):
        name = self._next_name()
        self.faces.append({
            "name": name,
            "embeddings": [embedding.flatten().tolist()],
            "sightings": [],
        })
        idx = len(self.faces) - 1
        self._avg_cache[idx] = embedding.reshape(1, -1).copy()
        self._last_embed_time[idx] = time.monotonic()
        self._dirty = True
        self.save_force()
        return name

    def add_sighting(self, name, start_dt, end_dt):
        """Record a completed sighting for the given person."""
        duration = (end_dt - start_dt).total_seconds()
        if duration < 0.5:
            return
        for face in self.faces:
            if face["name"] == name:
                face["sightings"].append({
                    "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": round(duration, 1),
                })
                self._dirty = True
                return

    def _next_name(self):
        idx = self.next_name_idx
        name = NAMES[idx % len(NAMES)]
        if idx >= len(NAMES):
            name += f" {idx // len(NAMES) + 1}"
        self.next_name_idx += 1
        return name


IOU_THRESHOLD = 0.3
PENDING_FRAMES = 20  # frames before registering a new face (~0.7s at 30fps)
TRACK_VERIFY_THRESHOLD = 0.25  # embedding sanity check after IoU match
RECOVERY_THRESHOLD = 0.35      # re-link lost tracks by embedding
MATCH_MARGIN = 0.05            # reject ambiguous DB matches
MIN_FACE_SIZE = 60             # skip tiny/partial faces (pixels)
STALE_FRAMES = 15  # remove tracks not seen for this many frames


class FaceTracker:
    """Frame-to-frame face tracker using bounding box IoU."""

    def __init__(self, db):
        self.tracks = {}
        self._next_id = 0
        self._db = db
        self.events = []  # (type, name) — "new", "enter", "exit"

    def clear(self):
        """Clear all tracks without recording sightings (used on DB clear)."""
        self.tracks.clear()

    def finalize_all(self):
        """End all active confirmed tracks and record their sightings."""
        now = datetime.now()
        for t in self.tracks.values():
            if t["confirmed"] and t.get("start_time"):
                self._db.add_sighting(t["name"], t["start_time"], now)

    def get_active_names(self):
        """Return set of names currently being tracked (confirmed only)."""
        return {t["name"] for t in self.tracks.values() if t["confirmed"]}

    def get_active_start_times(self):
        """Return dict of {name: start_time} for active confirmed tracks."""
        return {t["name"]: t["start_time"] for t in self.tracks.values()
                if t["confirmed"] and t.get("start_time")}

    def update(self, detections, frame, recognizer, db):
        """Match detections to tracks, return list of (bbox, name) for drawing."""
        results = []
        self.events.clear()
        if detections is None:
            # No faces: age all tracks
            self._age_tracks()
            return results

        det_list = []  # list of (face_array, bbox_tuple)
        for face in detections:
            x1, y1, fw, fh = face[:4].astype(int)
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue
            det_list.append((face, (x1, y1, x1 + fw, y1 + fh)))

        if not det_list:
            self._age_tracks()
            return results

        # Compute IoU matrix between detections and existing tracks
        track_ids = list(self.tracks.keys())
        matched_det = set()
        matched_trk = set()
        # Cache embeddings computed during matching for reuse later
        det_embeddings = {}

        if track_ids:
            track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]
            # Greedy assignment by best IoU
            iou_pairs = []
            for di, (_, dbbox) in enumerate(det_list):
                for ti, tbbox in enumerate(track_bboxes):
                    iou = self._iou(dbbox, tbbox)
                    if iou > IOU_THRESHOLD:
                        iou_pairs.append((iou, di, ti))
            iou_pairs.sort(reverse=True)

            for iou, di, ti in iou_pairs:
                if di in matched_det or ti in matched_trk:
                    continue
                tid = track_ids[ti]
                track = self.tracks[tid]
                face_arr, bbox = det_list[di]

                # Compute embedding
                aligned = recognizer.alignCrop(frame, face_arr)
                embedding = recognizer.feature(aligned)
                det_embeddings[di] = embedding

                # Embedding verification for confirmed tracks
                if track["confirmed"] and track["embedding"] is not None:
                    sim = recognizer.match(embedding, track["embedding"],
                                           cv2.FaceRecognizerSF_FR_COSINE)
                    if sim < TRACK_VERIFY_THRESHOLD:
                        # Different face in same position — reject IoU match
                        continue

                matched_det.add(di)
                matched_trk.add(ti)

                # Update track bbox
                track["bbox"] = bbox
                track["frames_seen"] += 1
                track["missed"] = 0
                track["embedding"] = embedding

                if track["confirmed"]:
                    # Add embedding to DB for confirmed tracks
                    match_idx = self._find_db_idx(db, track["name"])
                    if match_idx >= 0:
                        db.add_embedding(match_idx, embedding, recognizer)
                    results.append((bbox, track["name"], face_arr[14]))
                else:
                    # Pending track: accumulate embeddings
                    track["pending_embeddings"].append(embedding.copy())
                    if track["frames_seen"] >= PENDING_FRAMES:
                        # Promote with outlier-filtered averaging
                        avg_emb = self._filtered_average(
                            track["pending_embeddings"], recognizer)
                        # Re-check DB before registering
                        match_idx, _ = db.find_match(recognizer, avg_emb)
                        if match_idx >= 0:
                            track["name"] = db.faces[match_idx]["name"]
                            db.add_embedding(match_idx, avg_emb, recognizer)
                            self.events.append(("enter", track["name"]))
                            db.save_thumb(track["name"], frame, bbox)
                        else:
                            track["name"] = db.register(avg_emb)
                            print(f"New face registered: {track['name']}")
                            self.events.append(("new", track["name"]))
                            db.save_thumb(track["name"], frame, bbox)
                        track["confirmed"] = True
                        track["start_time"] = datetime.now()
                        track["pending_embeddings"] = []
                    results.append((bbox, track["name"], face_arr[14]))

        # Embedding-based track recovery: re-link unmatched detections
        # to unmatched confirmed tracks that were recently lost
        unmatched_det_ids = [di for di in range(len(det_list))
                            if di not in matched_det]
        unmatched_trk_ids = [ti for ti, tid in enumerate(track_ids)
                             if ti not in matched_trk
                             and self.tracks[tid]["confirmed"]
                             and self.tracks[tid]["missed"] < 5]
        if unmatched_det_ids and unmatched_trk_ids:
            recovery_pairs = []
            for di in unmatched_det_ids:
                if di not in det_embeddings:
                    face_arr, bbox = det_list[di]
                    aligned = recognizer.alignCrop(frame, face_arr)
                    det_embeddings[di] = recognizer.feature(aligned)
                emb = det_embeddings[di]
                for ti in unmatched_trk_ids:
                    tid = track_ids[ti]
                    track = self.tracks[tid]
                    if track["embedding"] is not None:
                        sim = recognizer.match(
                            emb, track["embedding"],
                            cv2.FaceRecognizerSF_FR_COSINE)
                        if sim >= RECOVERY_THRESHOLD:
                            recovery_pairs.append((sim, di, ti))
            recovery_pairs.sort(reverse=True)
            for sim, di, ti in recovery_pairs:
                if di in matched_det or ti in matched_trk:
                    continue
                matched_det.add(di)
                matched_trk.add(ti)
                tid = track_ids[ti]
                track = self.tracks[tid]
                face_arr, bbox = det_list[di]
                embedding = det_embeddings[di]
                track["bbox"] = bbox
                track["frames_seen"] += 1
                track["missed"] = 0
                track["embedding"] = embedding
                match_idx = self._find_db_idx(db, track["name"])
                if match_idx >= 0:
                    db.add_embedding(match_idx, embedding, recognizer)
                results.append((bbox, track["name"], face_arr[14]))

        # Unmatched detections: try DB match or create pending track
        for di, (face_arr, bbox) in enumerate(det_list):
            if di in matched_det:
                continue

            if di in det_embeddings:
                embedding = det_embeddings[di]
            else:
                aligned = recognizer.alignCrop(frame, face_arr)
                embedding = recognizer.feature(aligned)

            match_idx, _ = db.find_match(recognizer, embedding)
            if match_idx >= 0:
                name = db.faces[match_idx]["name"]
                db.add_embedding(match_idx, embedding, recognizer)
                self.events.append(("enter", name))
                db.save_thumb(name, frame, bbox)
                # Create confirmed track
                tid = self._new_id()
                self.tracks[tid] = {
                    "name": name,
                    "bbox": bbox,
                    "embedding": embedding,
                    "frames_seen": 1,
                    "missed": 0,
                    "confirmed": True,
                    "start_time": datetime.now(),
                    "pending_embeddings": [],
                }
                results.append((bbox, name, face_arr[14]))
            else:
                # Create pending track
                tid = self._new_id()
                self.tracks[tid] = {
                    "name": "?",
                    "bbox": bbox,
                    "embedding": embedding,
                    "frames_seen": 1,
                    "missed": 0,
                    "confirmed": False,
                    "start_time": None,
                    "pending_embeddings": [embedding.copy()],
                }
                results.append((bbox, "?", face_arr[14]))

        # Age unmatched tracks
        for ti, tid in enumerate(track_ids):
            if ti not in matched_trk:
                self.tracks[tid]["missed"] += 1

        # Remove stale tracks, record sightings for confirmed ones
        stale = [tid for tid, t in self.tracks.items() if t["missed"] > STALE_FRAMES]
        now = datetime.now()
        for tid in stale:
            t = self.tracks[tid]
            if t["confirmed"] and t.get("start_time"):
                self._db.add_sighting(t["name"], t["start_time"], now)
                self.events.append(("exit", t["name"]))
            del self.tracks[tid]

        return results

    def _age_tracks(self):
        """Increment missed counter for all tracks and remove stale ones."""
        for t in self.tracks.values():
            t["missed"] += 1
        stale = [tid for tid, t in self.tracks.items() if t["missed"] > STALE_FRAMES]
        now = datetime.now()
        for tid in stale:
            t = self.tracks[tid]
            if t["confirmed"] and t.get("start_time"):
                self._db.add_sighting(t["name"], t["start_time"], now)
                self.events.append(("exit", t["name"]))
            del self.tracks[tid]

    def _find_db_idx(self, db, name):
        for i, face in enumerate(db.faces):
            if face["name"] == name:
                return i
        return -1

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def _filtered_average(embeddings, recognizer):
        """Compute outlier-filtered average of pending embeddings."""
        if len(embeddings) < 4:
            return np.mean(embeddings, axis=0).reshape(1, -1)
        # Compute pairwise cosine similarity matrix
        n = len(embeddings)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                ei = embeddings[i].reshape(1, -1)
                ej = embeddings[j].reshape(1, -1)
                s = recognizer.match(ei, ej, cv2.FaceRecognizerSF_FR_COSINE)
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s
            sim_matrix[i, i] = 1.0
        # Average similarity of each embedding to all others
        avg_sims = sim_matrix.mean(axis=1)
        median_sim = np.median(avg_sims)
        # Keep embeddings with above-median average similarity
        keep = [i for i in range(n) if avg_sims[i] >= median_sim]
        if len(keep) < 3:
            keep = list(range(n))  # fallback to all
        filtered = [embeddings[i] for i in keep]
        return np.mean(filtered, axis=0).reshape(1, -1)

    @staticmethod
    def _iou(a, b):
        """Compute IoU between two (x1, y1, x2, y2) bboxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)


def format_duration(seconds):
    """Format duration in seconds to a human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


def load_models():
    for path, desc in [(YUNET_MODEL, "YuNet face detection"), (SFACE_MODEL, "SFace recognition")]:
        if not os.path.isfile(path):
            print(f"Error: {desc} model not found at '{path}'.")
            print("Run ./run.sh to download models automatically.")
            sys.exit(1)
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (320, 320),
        score_threshold=0.3, nms_threshold=0.3, top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    return detector, recognizer


def main():
    detector, recognizer = load_models()
    db = FaceDB(FACES_DB_FILE)
    tracker = FaceTracker(db)
    print(f"Loaded face database: {len(db.faces)} known face(s).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check that it's connected and you have permissions.")
        print("Tip: try running 'ls /dev/video*' to verify the device exists.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold %", WINDOW_NAME, DEFAULT_THRESHOLD, 100, lambda _: None)

    # Panel UI state
    expanded = set()          # face indices with expanded sighting list
    panel_scroll = [0]        # scroll offset (mutable for callback)
    frame_w = [640]           # current video width (mutable for callback)
    panel_rows = []           # (y_start, y_end, face_idx or None) for click detection

    def on_mouse(event, x, y, flags, _param):
        if x < frame_w[0]:
            return
        panel_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            for ry1, ry2, fidx in panel_rows:
                if fidx is not None and ry1 <= panel_y < ry2:
                    expanded.symmetric_difference_update({fidx})
                    break
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                panel_scroll[0] = max(0, panel_scroll[0] - 30)
            else:
                panel_scroll[0] += 30

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Webcam opened with face detection and recognition.")
    print("Press 'q' to quit, 'c' to clear, 'l' to switch language (EN/RU).")
    print("Use the slider to adjust the detection confidence threshold.")
    print("Click person names in the side panel to view sighting history.")

    save_timer = time.monotonic()
    toasts = []  # (text, expire_time, color, font_scale)
    lang = "en"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        h, w = frame.shape[:2]
        frame_w[0] = w
        detector.setInputSize((w, h))
        threshold = cv2.getTrackbarPos("Threshold %", WINDOW_NAME)

        _, detections = detector.detect(frame)

        # Filter detections by confidence threshold
        filtered = None
        if detections is not None:
            mask = detections[:, 14] >= threshold / 100.0
            if mask.any():
                filtered = detections[mask]

        # Run tracker
        tracked_faces = tracker.update(filtered, frame, recognizer, db)

        # Process tracker events into toasts
        now_mono = time.monotonic()
        for evt_type, evt_name in tracker.events:
            if evt_type == "new":
                toasts.append((tr(lang, "welcome", name=evt_name),
                               now_mono + TOAST_WELCOME_SEC, (0, 255, 255), 0.9))
            elif evt_type == "enter":
                toasts.append((tr(lang, "arrived", name=evt_name),
                               now_mono + TOAST_EVENT_SEC, (0, 255, 0), 0.7))
            elif evt_type == "exit":
                toasts.append((tr(lang, "left", name=evt_name),
                               now_mono + TOAST_EVENT_SEC, (0, 150, 255), 0.7))
        toasts = [t for t in toasts if t[1] > now_mono]

        for bbox, name, confidence in tracked_faces:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 255) if name == "?" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {confidence:.0%}"
            label_y = y1 - 10
            if label_y < 20:
                label_y = y2 + 20
            label_x = max(x1, 0)
            cv2.putText(frame, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        hud_text = tr(lang, "hud", t=threshold, n=len(db.faces))
        draw_text(frame, hud_text, (10, 25), 0.7, (255, 255, 255), 2)

        # Render toasts (bottom of frame, stacked upward)
        toast_y = h - 40
        for ttext, _expire, tcolor, tscale in reversed(toasts):
            tw, th = text_size(ttext, tscale, 2)
            tx = (w - tw) // 2
            draw_text(frame, ttext, (tx, toast_y), tscale, (0, 0, 0), 4)
            draw_text(frame, ttext, (tx, toast_y), tscale, tcolor, 2)
            toast_y -= th + 15

        # --- Side panel ---
        panel = np.full((h, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, h), (80, 80, 80), 1)
        draw_text(panel, tr(lang, "panel_header"), (10, 25), 0.7, (255, 255, 255), 2)
        cv2.line(panel, (5, PANEL_HEADER_H), (PANEL_WIDTH - 5, PANEL_HEADER_H), (80, 80, 80), 1)

        active_names = tracker.get_active_names()
        active_starts = tracker.get_active_start_times()
        new_rows = []
        y = PANEL_HEADER_H + 5 - panel_scroll[0]

        for fi, face in enumerate(db.faces):
            name = face["name"]
            sightings = face.get("sightings", [])
            is_active = name in active_names
            is_expanded = fi in expanded

            if PANEL_HEADER_H < y < h:
                # Expand/collapse triangle
                tx, ty = 10, y + 10
                if is_expanded:
                    pts = np.array([[tx, ty], [tx + 10, ty], [tx + 5, ty + 8]], np.int32)
                else:
                    pts = np.array([[tx, ty], [tx + 8, ty + 5], [tx, ty + 10]], np.int32)
                cv2.fillPoly(panel, [pts], (180, 180, 180))

                # Name
                name_color = (0, 255, 0) if is_active else (200, 200, 200)
                cv2.putText(panel, name, (25, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, name_color, 1)

                # Status + count
                n_sightings = len(sightings) + (1 if is_active else 0)
                tag_key = "live" if is_active else "count"
                tag = tr(lang, tag_key, c=n_sightings)
                tag_color = (0, 255, 0) if is_active else (150, 150, 150)
                draw_text(panel, tag, (PANEL_WIDTH - 100, y + 18), 0.4, tag_color, 1)
                new_rows.append((y, y + ROW_HEIGHT, fi))
            y += ROW_HEIGHT

            # Expanded: show thumbnail + sighting entries
            if is_expanded:
                # Show thumbnail
                thumb = db.get_thumb(name)
                if thumb is not None:
                    ty1 = y + 2
                    ty2 = ty1 + THUMB_SIZE
                    tx1, tx2 = 20, 20 + THUMB_SIZE
                    if PANEL_HEADER_H < ty1 and ty2 < h:
                        panel[ty1:ty2, tx1:tx2] = thumb
                    y += THUMB_SIZE + 4

                # Show current live session first
                if is_active and name in active_starts:
                    if PANEL_HEADER_H < y < h:
                        dur = (datetime.now() - active_starts[name]).total_seconds()
                        line = tr(lang, "now", d=format_duration(dur))
                        draw_text(panel, line, (20, y + 14), 0.35, (0, 200, 255), 1)
                        new_rows.append((y, y + SIGHTING_ROW_H, None))
                    y += SIGHTING_ROW_H

                # Past sightings (most recent first, last 15)
                for s in reversed(sightings[-15:]):
                    if PANEL_HEADER_H < y < h:
                        dur = format_duration(s["duration_sec"])
                        try:
                            dt = datetime.strptime(s["start"], "%Y-%m-%d %H:%M:%S")
                            display = dt.strftime("%b %d %H:%M")
                        except ValueError:
                            display = s["start"][:16]
                        line = f"  {display}  {dur}"
                        cv2.putText(panel, line, (20, y + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 180, 140), 1)
                        new_rows.append((y, y + SIGHTING_ROW_H, None))
                    y += SIGHTING_ROW_H

            # Separator
            if PANEL_HEADER_H < y < h:
                cv2.line(panel, (5, y), (PANEL_WIDTH - 5, y), (50, 50, 50), 1)
            y += 3

        panel_rows.clear()
        panel_rows.extend(new_rows)

        canvas = np.hstack([frame, panel])
        cv2.imshow(WINDOW_NAME, canvas)

        now = time.monotonic()
        if now - save_timer > 10.0:
            db.save()
            save_timer = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            lang = "ru" if lang == "en" else "en"
        if key == ord("c"):
            tracker.clear()
            db.clear()
            expanded.clear()
            panel_scroll[0] = 0
            toasts.clear()
            print(tr(lang, "db_cleared"))
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    tracker.finalize_all()
    db.save_force()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Database saved: {len(db.faces)} face(s).")


if __name__ == "__main__":
    main()
