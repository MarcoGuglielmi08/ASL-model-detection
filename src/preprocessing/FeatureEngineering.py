import pandas as pd
import numpy as np

from src.util.paths import DATA_DIR


# Build the tabular feature set used by the classifiers.
class FeatureEngineering:
    INPUT_CSV = DATA_DIR / "asl_landmarks_clean_level3.csv"
    OUTPUT_CSV = DATA_DIR / "asl_features_engineered.csv"

    LANDMARK_START_COL = 7
    USE_Z = False

    # MediaPipe landmark indices used to derive geometric features.
    WRIST = 0

    FINGERS = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }

    FINGERTIPS = {
        "thumb":  4,
        "index":  8,
        "middle": 12,
        "ring":   16,
        "pinky":  20,
    }

    def get_points(self, row):
        # Return landmarks as 2D or 3D points depending on USE_Z.
        landmarks = pd.to_numeric(
            row.iloc[self.LANDMARK_START_COL:],
            errors="coerce"
        ).values.astype(float)

        points = landmarks.reshape(21, 3)

        if not self.USE_Z:
            points = points[:, :2]

        return points

    def euclidean(self, a, b):
        # Compute distance between two landmark points.
        return np.linalg.norm(a - b)

    def compute_angle(self, a, b, c):
        # Compute the angle at joint b formed by points a-b-c.
        ba = a - b
        bc = c - b
        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator < 1e-8:
            return 0.0
        cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
        return np.arccos(cosine)

    def extract_distance_features(self, points):
        # Create fingertip-to-fingertip and wrist-to-fingertip distances.
        features = {}

        fingertip_pairs = [
            ("thumb", "index"), ("thumb", "middle"),
            ("thumb", "ring"),  ("thumb", "pinky"),
            ("index", "middle"), ("middle", "ring"), ("ring", "pinky"),
        ]

        for a, b in fingertip_pairs:
            features[f"{a}_{b}_tip_distance"] = self.euclidean(
                points[self.FINGERTIPS[a]], points[self.FINGERTIPS[b]]
            )

        wrist = points[self.WRIST]
        for finger, tip_idx in self.FINGERTIPS.items():
            features[f"wrist_to_{finger}_tip"] = self.euclidean(wrist, points[tip_idx])

        return features

    def extract_angle_features(self, points):
        # Create two joint-angle features for each finger.
        features = {}

        for finger_name, joints in self.FINGERS.items():
            features[f"{finger_name}_angle_1"] = self.compute_angle(
                points[joints[0]], points[joints[1]], points[joints[2]]
            )
            features[f"{finger_name}_angle_2"] = self.compute_angle(
                points[joints[1]], points[joints[2]], points[joints[3]]
            )

        return features

    def extract_vector_features(self, points):
        # Create wrist-to-fingertip direction vectors.
        features = {}
        wrist = points[self.WRIST]

        for finger, tip_idx in self.FINGERTIPS.items():
            vec = points[tip_idx] - wrist
            features[f"{finger}_vector_x"] = vec[0]
            features[f"{finger}_vector_y"] = vec[1]
            if self.USE_Z:
                features[f"{finger}_vector_z"] = vec[2]

        return features

    def run(self):
        print("Running feature engineering...")

        df = pd.read_csv(self.INPUT_CSV)
        print(f"Input rows: {len(df)}")

        engineered_rows = []

        for idx, row in df.iterrows():
            try:
                points = self.get_points(row)

                features = {
                    "filepath":   row["filepath"],
                    "label":      row["label"],
                    "handedness": row["handedness"],
                }

                landmark_values = pd.to_numeric(
                    row.iloc[self.LANDMARK_START_COL:],
                    errors="coerce"
                ).values.astype(float)

                for i, value in enumerate(landmark_values):
                    features[f"lm_{i}"] = value

                # Keep raw normalized landmarks and append compact geometry.
                features.update(self.extract_distance_features(points))
                features.update(self.extract_angle_features(points))
                features.update(self.extract_vector_features(points))

                engineered_rows.append(features)

                if (idx + 1) % 1000 == 0:
                    print(f"Processed: {idx + 1}")

            except Exception as e:
                print(f"[ERROR] Row {idx}: {e}")

        engineered_df = pd.DataFrame(engineered_rows)
        engineered_df.to_csv(self.OUTPUT_CSV, index=False)

        print("Feature engineering complete.")
        print(f"Output rows: {len(engineered_df)}")
        print(f"Saved to: {self.OUTPUT_CSV}")
