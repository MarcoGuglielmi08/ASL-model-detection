import csv
import math

import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.utils.paths import DATA_DIR, ASSETS_DIR


# Extract one normalized MediaPipe hand landmark vector for each image.
class MediapipeLandmarksExtractor:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    DATASET_DIR = DATA_DIR / "ASL_Alphabet_Dataset" / "asl_alphabet_train"

    OUTPUT_CSV = DATA_DIR / "asl_landmarks_mediapipe.csv"

    FAILED_CSV = DATA_DIR / "asl_landmarks_failed.csv"

    MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"

    LIMIT = None
    MAX_IMAGE_SIZE = 640

    def __init__(self):
        self.dataset_dir = self.DATASET_DIR.resolve()

    def get_landmark_columns(self):
        # Return the 63 raw landmark feature names: x/y/z for 21 points.
        columns = []
        for i in range(21):
            columns += [f"x{i}", f"y{i}", f"z{i}"]
        return columns

    def get_label(self, image_path):
        # Infer the ASL label from the image parent folder.
        relative_path = image_path.relative_to(self.dataset_dir)
        return relative_path.parts[0]

    def resize_if_needed(self, image, max_size=640):
        # Downscale large images to keep MediaPipe inference stable and faster.
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def apply_clahe(self, image_bgr):
        # Improve local contrast for images where the first detection fails.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def normalize_landmarks(self, landmarks):
        # Center landmarks on the wrist and scale them by hand size.
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        dz = middle_mcp.z - wrist.z

        scale = math.sqrt(dx * dx + dy * dy + dz * dz)
        if scale == 0:
            scale = 1

        normalized = []
        for point in landmarks:
            normalized.extend([
                (point.x - wrist.x) / scale,
                (point.y - wrist.y) / scale,
                (point.z - wrist.z) / scale,
            ])
        return normalized

    def detect_hand(self, landmarker, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return landmarker.detect(mp_image)

    def process_detection(self, landmarker, image_bgr):
        # Try the original image, a horizontal flip, and contrast enhancement.
        image_bgr = self.resize_if_needed(image_bgr, self.MAX_IMAGE_SIZE)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.detect_hand(landmarker, image_rgb)
        if len(result.hand_landmarks) == 1:
            return result

        flipped = cv2.flip(image_rgb, 1)
        result = self.detect_hand(landmarker, flipped)
        if len(result.hand_landmarks) == 1:
            return result

        clahe_bgr = self.apply_clahe(image_bgr)
        clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
        result = self.detect_hand(landmarker, clahe_rgb)
        if len(result.hand_landmarks) == 1:
            return result

        return result

    def run(self):
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {self.MODEL_PATH}")

        self.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        self.FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        image_paths = [
            path
            for path in sorted(self.dataset_dir.rglob("*"))
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        if self.LIMIT:
            image_paths = image_paths[:self.LIMIT]

        if not image_paths:
            raise ValueError(f"No images found in dataset directory: {self.dataset_dir}")

        columns = [
            "filepath", "label", "handedness", "score",
            "detected_hands", "image_width", "image_height",
            *self.get_landmark_columns(),
        ]

        with (
            vision.HandLandmarker.create_from_options(options) as landmarker,
            self.OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file,
            self.FAILED_CSV.open("w", newline="", encoding="utf-8") as failed_file,
        ):
            writer = csv.writer(output_file)
            failed_writer = csv.writer(failed_file)

            writer.writerow(columns)
            failed_writer.writerow(["filepath", "label", "reason", "detected_hands"])

            saved = 0
            failed = 0
            current_label = None

            for index, image_path in enumerate(image_paths, start=1):
                label = self.get_label(image_path)

                if label != current_label:
                    current_label = label
                    print(f"Processing letter: {label}")

                filepath = str(image_path.resolve())

                try:
                    image_bgr = cv2.imread(str(image_path))

                    if image_bgr is None:
                        failed += 1
                        failed_writer.writerow([filepath, label, "unreadable_image", 0])
                        continue

                    height, width = image_bgr.shape[:2]

                    result = self.process_detection(landmarker, image_bgr)
                    detected_hands = len(result.hand_landmarks)

                    if detected_hands == 0:
                        failed += 1
                        failed_writer.writerow([filepath, label, "no_hand_detected", 0])
                        continue

                    if detected_hands > 1:
                        failed += 1
                        failed_writer.writerow([filepath, label, "multiple_hands_detected", detected_hands])
                        continue

                    landmarks = result.hand_landmarks[0]
                    handedness = result.handedness[0][0]
                    normalized_landmarks = self.normalize_landmarks(landmarks)

                    writer.writerow([
                        filepath, label, handedness.category_name,
                        handedness.score, detected_hands, width, height,
                        *normalized_landmarks,
                    ])
                    saved += 1

                    if index % 1000 == 0:
                        print(f"Processed: {index} | Saved: {saved} | Failed: {failed}")

                except Exception as e:
                    failed += 1
                    failed_writer.writerow([filepath, label, str(e), 0])
                    print(f"Error processing: {filepath}\n{e}")

        print(f"Extraction complete. Saved: {saved} | Failed: {failed}")
        print(f"CSV landmarks: {self.OUTPUT_CSV}")
        print(f"CSV failed:    {self.FAILED_CSV}")


if __name__ == "__main__":
    MediapipeLandmarksExtractor().run()
