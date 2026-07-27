import pandas as pd
import numpy as np

from src.util.paths import DATA_DIR


# Second cleaning stage: remove geometrically implausible detections.
class GeometricOutlierCheck:
    INPUT_CSV = DATA_DIR / "asl_landmarks_clean_level1.csv"
    OUTPUT_CSV = DATA_DIR / "asl_landmarks_clean_level2.csv"
    FAILED_CSV = DATA_DIR / "asl_landmarks_failed_level2.csv"

    LANDMARK_START_COL = 7
    EPS = 1e-6

    VAR_THRESHOLD       = 0.0005
    BBOX_THRESHOLD      = 0.02
    FINGERTIP_THRESHOLD = 0.01

    FINGERTIPS = [4, 8, 12, 16, 20]  # thumb to pinky tips

    def run(self):
        df = pd.read_csv(self.INPUT_CSV)

        print("Running geometric consistency check...")
        print(f"Input rows: {len(df)}")

        valid_rows  = []
        failed_rows = []

        for idx, row in df.iterrows():
            reasons = []

            landmarks = pd.to_numeric(
                row.iloc[self.LANDMARK_START_COL:],
                errors="coerce"
            ).values.astype(float)

            if np.any(np.isnan(landmarks)):
                reasons.append("nan_detected")
                failed_rows.append({"filepath": row["filepath"], "label": row["label"], "reasons": ";".join(reasons)})
                continue

            points = landmarks.reshape(-1, 3)

            # Very low variance means the landmarks are nearly collapsed.
            var = np.var(points)
            if var < self.VAR_THRESHOLD:
                reasons.append(f"low_variance:{var:.6f}")

            # A tiny bounding box is unlikely to describe a usable hand pose.
            bbox = np.max(points, axis=0) - np.min(points, axis=0)
            bbox_size = np.linalg.norm(bbox)
            if bbox_size < self.BBOX_THRESHOLD:
                reasons.append(f"small_bbox:{bbox_size:.6f}")

            # Detect the extreme case where every point overlaps the wrist.
            if np.all(np.linalg.norm(points - points[0], axis=1) < self.EPS):
                reasons.append("collapsed_landmarks")

            # Fingertips should be separated from the wrist in a valid hand.
            wrist = points[0]
            fingertip_dists = [np.linalg.norm(points[i] - wrist) for i in self.FINGERTIPS]
            mean_finger_dist = np.mean(fingertip_dists)

            if mean_finger_dist < self.FINGERTIP_THRESHOLD:
                reasons.append(f"tiny_fingers:{mean_finger_dist:.6f}")

            if len(reasons) == 0:
                valid_rows.append(row)
            else:
                failed_rows.append({
                    "filepath": row["filepath"],
                    "label":    row["label"],
                    "reasons":  ";".join(reasons),
                })
                print(f"[DROP] {row['filepath']} -> {reasons}")

        clean_df  = pd.DataFrame(valid_rows)
        failed_df = pd.DataFrame(failed_rows)

        print("Geometric consistency summary:")
        print(f"Input rows:   {len(df)}")
        print(f"Valid rows:   {len(clean_df)}")
        print(f"Dropped rows: {len(failed_df)}")

        if len(failed_df) > 0:
            print("\nDrop reason distribution:")
            print(failed_df["reasons"].value_counts())

        clean_df.to_csv(self.OUTPUT_CSV,  index=False)
        failed_df.to_csv(self.FAILED_CSV, index=False)

        print(f"Clean:  {self.OUTPUT_CSV}")
        print(f"Failed: {self.FAILED_CSV}")
