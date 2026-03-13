"""Webcam Viewer — Phase 3: Live video feed with face detection and recognition."""

import os
import sys
import json
import time
import numpy as np
import cv2


MODEL_DIR = "model"
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
FACES_DB_FILE = "faces_db.json"
DEFAULT_THRESHOLD = 60
COSINE_THRESHOLD = 0.4
WINDOW_NAME = "Webcam Viewer"
MAX_EMBEDDINGS = 20
EMBEDDING_INTERVAL = 2.0  # seconds between storing embeddings for a known face

NAMES = [
    "Atlas", "Nova", "Echo", "Sage", "Orion",
    "Luna", "Blaze", "Coral", "Drift", "Ember",
    "Frost", "Haze", "Iris", "Jade", "Kai",
    "Lyra", "Mars", "Nyx", "Onyx", "Pearl",
    "Quinn", "Rune", "Sol", "Thorn", "Uma",
    "Vale", "Wren", "Xena", "Yara", "Zephyr",
]


class FaceDB:
    """Persistent face database stored as JSON."""

    def __init__(self, path):
        self.path = path
        self.faces = []
        self.next_name_idx = 0
        self._dirty = False
        self._avg_cache = {}
        self._last_embed_time = {}
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.faces = data.get("faces", [])
            self.next_name_idx = data.get("next_name_idx", 0)
        self._rebuild_cache()

    def _rebuild_cache(self):
        self._avg_cache.clear()
        for i, face in enumerate(self.faces):
            if face["embeddings"]:
                avg = np.array(face["embeddings"], dtype=np.float32).mean(axis=0).reshape(1, -1)
                self._avg_cache[i] = avg

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
        self.save_force()

    def find_match(self, recognizer, embedding):
        best_score = -1
        best_idx = -1
        for i, avg in self._avg_cache.items():
            score = recognizer.match(embedding, avg, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score >= COSINE_THRESHOLD:
            return best_idx, best_score
        return -1, best_score

    def add_embedding(self, face_idx, embedding):
        now = time.monotonic()
        last = self._last_embed_time.get(face_idx, 0)
        if now - last < EMBEDDING_INTERVAL:
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
        })
        idx = len(self.faces) - 1
        self._avg_cache[idx] = embedding.reshape(1, -1).copy()
        self._last_embed_time[idx] = time.monotonic()
        self._dirty = True
        self.save_force()
        return name

    def _next_name(self):
        idx = self.next_name_idx
        name = NAMES[idx % len(NAMES)]
        if idx >= len(NAMES):
            name += f" {idx // len(NAMES) + 1}"
        self.next_name_idx += 1
        return name


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

    print("Webcam opened with face detection and recognition.")
    print("Press 'q' to quit, 'c' to clear the face database.")
    print("Use the slider to adjust the detection confidence threshold.")

    save_timer = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        threshold = cv2.getTrackbarPos("Threshold %", WINDOW_NAME)

        _, detections = detector.detect(frame)

        if detections is not None:
            for face in detections:
                confidence = face[14]
                if confidence < threshold / 100.0:
                    continue

                x1, y1, fw, fh = face[:4].astype(int)
                x2, y2 = x1 + fw, y1 + fh

                aligned = recognizer.alignCrop(frame, face)
                embedding = recognizer.feature(aligned)

                match_idx, _ = db.find_match(recognizer, embedding)

                if match_idx >= 0:
                    name = db.faces[match_idx]["name"]
                    db.add_embedding(match_idx, embedding)
                else:
                    name = db.register(embedding)
                    print(f"New face registered: {name}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {confidence:.0%}"
                label_y = y1 - 10
                if label_y < 20:
                    label_y = y2 + 20
                label_x = max(x1, 0)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Threshold: {threshold}%  |  Known: {len(db.faces)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        now = time.monotonic()
        if now - save_timer > 10.0:
            db.save()
            save_timer = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            db.clear()
            print("Face database cleared.")
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    db.save_force()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Database saved: {len(db.faces)} face(s).")


if __name__ == "__main__":
    main()
