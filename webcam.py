"""Webcam Viewer — Phase 1: Live video feed display."""

import sys
import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Check that it's connected and you have permissions.")
        print("Tip: try running 'ls /dev/video*' to verify the device exists.")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        cv2.imshow("Webcam Viewer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Webcam Viewer", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
