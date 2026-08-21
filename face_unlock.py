from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorflow as tf


THRESHOLD = 0.85
CAMERA_INDEX = 0
COUNTDOWN_SECONDS = 5


def app_path(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def load_face_detector():
    detector_path = app_path("face_detection_yunet_2023mar.onnx")
    if not detector_path.exists():
        raise FileNotFoundError(f"Missing face detector: {detector_path}")
    return cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.8, 0.3, 5000)


def load_face(image, detector, target_size=(224, 224)):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(image)
    if detections is None or len(detections) == 0:
        raise ValueError("No face detected. Improve lighting and face the camera.")

    detection = max(detections, key=lambda item: item[2] * item[3])
    x, y, box_width, box_height = (int(value) for value in detection[:4])
    padding_x, padding_y = int(box_width * 0.15), int(box_height * 0.15)
    x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
    x2 = min(width, x + box_width + padding_x)
    y2 = min(height, y + box_height + padding_y)
    face = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return cv2.resize(face, target_size), (x, y, box_width, box_height)


def embedding(image, detector, model):
    face, box = load_face(image, detector)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(
        face.astype(np.float32)[None, ...]
    )
    vector = model.predict(batch, verbose=0)[0]
    vector /= np.linalg.norm(vector) + 1e-8
    return vector, box


def capture_and_verify(enrollment, detector, model):
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check Windows camera permissions.")

    window_name = "Face Unlock - look at camera"
    started_at = time.monotonic()
    last_box = None
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The webcam did not provide a frame.")

            preview = frame.copy()
            try:
                _, last_box = load_face(frame, detector)
                x, y, box_width, box_height = last_box
                cv2.rectangle(preview, (x, y), (x + box_width, y + box_height), (0, 255, 0), 3)
                face_status = "Face detected"
            except ValueError:
                last_box = None
                face_status = "Face not detected"

            seconds_left = max(0, COUNTDOWN_SECONDS - int(time.monotonic() - started_at))
            cv2.putText(preview, f"{face_status} | Capture in {seconds_left}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, "Press Space to capture | Q/Esc to cancel", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return False, None
            if (key == 32 or time.monotonic() - started_at >= COUNTDOWN_SECONDS) and last_box is not None:
                candidate, _ = embedding(frame, detector, model)
                similarity = float(np.dot(enrollment, candidate))
                return similarity >= THRESHOLD, similarity
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    enrollment_path = app_path("enrollment_embedding.npy")
    if not enrollment_path.exists():
        raise FileNotFoundError(f"Missing enrollment file: {enrollment_path}")

    print("Loading face model...")
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    detector = load_face_detector()
    enrollment = np.load(enrollment_path).astype(np.float32)
    enrollment /= np.linalg.norm(enrollment) + 1e-8
    unlocked, similarity = capture_and_verify(enrollment, detector, model)
    if similarity is None:
        print("Cancelled")
        return 1
    status = "UNLOCKED" if unlocked else "LOCKED"
    print(f"{status} | cosine similarity: {similarity:.3f} | threshold: {THRESHOLD:.3f}")
    return 0 if unlocked else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Face unlock failed: {error}")
        raise SystemExit(1)
