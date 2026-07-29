import pandas as pd
import numpy as np

from src.utils.paths import DATA_DIR


# Third cleaning stage: check anatomical consistency of hand structure.
class StructuralConsistencyCheck:
    INPUT_CSV = DATA_DIR / "asl_landmarks_clean_level2.csv"
    OUTPUT_CSV = DATA_DIR / "asl_landmarks_clean_level3.csv"
    FAILED_CSV = DATA_DIR / "asl_landmarks_failed_level3.csv"

    LANDMARK_START_COL = 7

    FINGERS = {
        "thumb":  [0, 1, 2, 3, 4],
        "index":  [0, 5, 6, 7, 8],
        "middle": [0, 9, 10, 11, 12],
        "ring":   [0, 13, 14, 15, 16],
        "pinky":  [0, 17, 18, 19, 20],
    }

    MIN_FINGER_RATIO = 0.2
    MAX_FINGER_RATIO = 2.5

    def get_points(self, row):
        # Return the 21 MediaPipe landmarks as a 21x3 numeric array.
        landmarks = pd.to_numeric(
            row.iloc[self.LANDMARK_START_COL:],
            errors="coerce"
        ).values.astype(float)
        return landmarks.reshape(-1, 3)

    def finger_length(self, points, finger):
        # Approximate a finger length by summing adjacent joint distances.
        pts = self.FINGERS[finger]
        length = 0.0
        for i in range(len(pts) - 1):
            length += np.linalg.norm(points[pts[i + 1]] - points[pts[i]])
        return length

    def run(self):
        df = pd.read_csv(self.INPUT_CSV)

        print("Running structural consistency check...")
        print(f"Input rows: {len(df)}")

        valid_rows  = []
        failed_rows = []

        for idx, row in df.iterrows():
            reasons = []
            points  = self.get_points(row)

            lengths = {f: self.finger_length(points, f) for f in self.FINGERS}

            # A valid hand should not contain fingers with near-zero length.
            if any(l < 1e-4 for l in lengths.values()):
                reasons.append("collapsed_finger_detected")

            # Extremely unbalanced finger lengths usually indicate bad landmarks.
            vals    = list(lengths.values())
            max_len = max(vals)
            min_len = min(vals)
            ratio   = (max_len / min_len) if min_len > 0 else 999

            if ratio < self.MIN_FINGER_RATIO or ratio > self.MAX_FINGER_RATIO:
                reasons.append(f"bad_finger_ratio:{ratio:.3f}")

            # The thumb and index should remain within a broad plausible ratio.
            thumb_index_ratio = lengths["thumb"] / (lengths["index"] + 1e-6)
            if thumb_index_ratio < 0.1 or thumb_index_ratio > 5:
                reasons.append(f"thumb_index_anomaly:{thumb_index_ratio:.3f}")

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

        print("Structural consistency summary:")
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
