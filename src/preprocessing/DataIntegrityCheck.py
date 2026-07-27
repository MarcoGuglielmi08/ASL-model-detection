import pandas as pd
import numpy as np

from src.util.paths import DATA_DIR


# First cleaning stage: remove rows with invalid landmark arrays.
class DataIntegrityCheck:
    INPUT_CSV = DATA_DIR / "asl_landmarks_mediapipe.csv"
    OUTPUT_CSV = DATA_DIR / "asl_landmarks_clean_level1.csv"
    FAILED_CSV = DATA_DIR / "asl_landmarks_failed_level1.csv"

    LANDMARK_START_COL = 7
    EXPECTED_LANDMARKS = 63  # 21 * 3

    def run(self):
        df = pd.read_csv(self.INPUT_CSV)

        print("Running data integrity check...")
        print(f"Initial rows: {len(df)}")

        valid_rows  = []
        failed_rows = []

        for idx, row in df.iterrows():
            reasons = []

            landmarks = pd.to_numeric(
                row.iloc[self.LANDMARK_START_COL:],
                errors="coerce"
            ).values.astype(float)

            # Invalid numeric values cannot be used by later geometry checks.
            if np.any(pd.isna(landmarks)) or np.any(np.isinf(landmarks)):
                reasons.append("nan_or_inf")

            # MediaPipe Hands returns 21 landmarks, each with x/y/z coordinates.
            if len(landmarks) != self.EXPECTED_LANDMARKS:
                reasons.append("wrong_feature_length")

            # Fully zero rows indicate a failed or empty extraction.
            if np.all(landmarks == 0):
                reasons.append("all_zero_landmarks")

            # If all points coincide, the detected hand has collapsed.
            reshaped = landmarks.reshape(-1, 3)
            if np.allclose(reshaped, reshaped[0]):
                reasons.append("collapsed_hand_landmarks")

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

        print("Data integrity summary:")
        print(f"Initial rows: {len(df)}")
        print(f"Valid rows:   {len(clean_df)}")
        print(f"Dropped rows: {len(failed_df)}")

        if len(failed_df) > 0:
            print("\nDrop reasons distribution:")
            print(failed_df["reasons"].value_counts())

        clean_df.to_csv(self.OUTPUT_CSV,  index=False)
        failed_df.to_csv(self.FAILED_CSV, index=False)

        print(f"Clean CSV:  {self.OUTPUT_CSV}")
        print(f"Failed CSV: {self.FAILED_CSV}")
