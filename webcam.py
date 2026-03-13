"""Webcam Viewer — Phase 2: Live video feed with face detection."""

import os
import sys
import cv2


MODEL_DIR = "model"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DEFAULT_THRESHOLD = 60
WINDOW_NAME = "Webcam Viewer"


def load_face_detector():
    if not os.path.isfile(PROTOTXT) or not os.path.isfile(CAFFEMODEL):
        print(f"Error: Face detection model not found in '{MODEL_DIR}/'.")
        print("Run ./run.sh to download it automatically.")
        sys.exit(1)
    return cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)


def detect_faces(net, frame, threshold):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold / 100.0:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            faces.append((x1, y1, x2, y2, confidence))
    return faces


def draw_faces(frame, faces):
    h, w = frame.shape[:2]
    for x1, y1, x2, y2, confidence in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{confidence:.0%}"
        label_y = y1 - 10
        if label_y < 20:
            label_y = y2 + 20
        label_x = max(x1, 0)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def on_threshold_change(_):
    pass


def main():
    net = load_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check that it's connected and you have permissions.")
        print("Tip: try running 'ls /dev/video*' to verify the device exists.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold %", WINDOW_NAME, DEFAULT_THRESHOLD, 100, on_threshold_change)

    print("Webcam opened with face detection. Press 'q' to quit.")
    print("Use the slider to adjust the confidence threshold.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        threshold = cv2.getTrackbarPos("Threshold %", WINDOW_NAME)
        faces = detect_faces(net, frame, threshold)
        draw_faces(frame, faces)

        cv2.putText(frame, f"Threshold: {threshold}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
