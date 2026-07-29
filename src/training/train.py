import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.utils.preprocessor import make_preprocessor
from src.utils.paths import DATA_DIR, MODELS_DIR
from utils.paths import NESTED_DIR


# Train the final classifier selected by nested cross-validation.
class Trainer:
    DATA_PATH = DATA_DIR / "asl_features_engineered.csv"
    RESULTS_PATH = NESTED_DIR / "group_nested_kfold_cv_results.csv"
    MODEL_OUTPUT_PATH = MODELS_DIR / "asl_model.pkl"

    TARGET_COLUMN = "label"
    DROP_COLUMNS  = ["label", "filepath", "handedness"]

    RANDOM_STATE = 42
    EXPECTED_OUTER_FOLDS = 5

    def _estimator_registry(self) -> dict:
        # Map each model name in the results CSV to its sklearn estimator.
        return {
            "KNN": lambda p: KNeighborsClassifier(
                n_neighbors = p.get("n_neighbors", 5),
                weights      = p.get("weights", "uniform"),
                p            = p.get("p", 2),
            ),

            "Decision Tree": lambda p: DecisionTreeClassifier(
                criterion          = p.get("criterion", "gini"),
                max_depth          = p.get("max_depth", None),
                min_samples_split  = p.get("min_samples_split", 2),
                min_samples_leaf   = p.get("min_samples_leaf", 1),
                random_state       = self.RANDOM_STATE,
            ),

            "Random Forest": lambda p: RandomForestClassifier(
                n_estimators      = p.get("n_estimators", 500),
                max_depth         = p.get("max_depth", None),
                max_features      = p.get("max_features", "sqrt"),
                min_samples_leaf  = p.get("min_samples_leaf", 1),
                bootstrap         = True,
                class_weight      = "balanced",
                random_state      = self.RANDOM_STATE,
                n_jobs            = -1,
            ),

            "HistGradientBoosting": lambda p: HistGradientBoostingClassifier(
                learning_rate     = p.get("learning_rate", 0.05),
                max_iter          = p.get("max_iter", 800),
                max_leaf_nodes    = p.get("max_leaf_nodes", 31),
                min_samples_leaf  = p.get("min_samples_leaf", 40),
                l2_regularization = p.get("l2_regularization", 0.0),
                early_stopping    = p.get("early_stopping", False),
                n_iter_no_change  = p.get("n_iter_no_change", 20),
                random_state      = self.RANDOM_STATE,
            ),

            "Approx RBF SVM": lambda p: Pipeline([
                ("rbf_features", Nystroem(
                    kernel       = "rbf",
                    n_components = p.get("rbf_features__n_components", 300),
                    gamma        = p.get("rbf_features__gamma", 0.001),
                    random_state = self.RANDOM_STATE,
                )),
                ("linear_svm", LinearSVC(
                    C            = p.get("linear_svm__C", 1.0),
                    class_weight = "balanced",
                    max_iter     = 10000,
                    random_state = self.RANDOM_STATE,
                )),
            ]),
        }

    def _load_best_result(self) -> tuple[str, dict, bool]:
        # Read the nested k-fold summary CSV and return:
        # (best_model_name, best_params_dict, use_scaler)
        if not self.RESULTS_PATH.exists():
            raise FileNotFoundError(
                f"Results CSV not found: {self.RESULTS_PATH}\n"
                "Run ModelNestedKFold first."
            )

        summary = pd.read_csv(self.RESULTS_PATH)

        required_columns = {"model", "outer_folds", "f1_macro_pct_mean"}
        missing_columns = required_columns - set(summary.columns)
        if missing_columns:
            raise ValueError(
                "Results CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        incomplete_rows = summary[summary["outer_folds"] < self.EXPECTED_OUTER_FOLDS]
        if not incomplete_rows.empty:
            details = incomplete_rows[["model", "outer_folds"]].to_string(index=False)
            raise ValueError(
                "Nested CV results are incomplete. "
                f"Expected {self.EXPECTED_OUTER_FOLDS} outer folds for each model.\n"
                f"{details}\n"
                "Re-run ModelNestedKFold before final training."
            )

        best_row        = summary.sort_values("f1_macro_pct_mean", ascending=False).iloc[0]
        best_model_name = best_row["model"]
        best_f1         = best_row["f1_macro_pct_mean"]

        print(f"\nBest model from nested CV: {best_model_name} (F1 macro mean: {best_f1:.2f}%)")
        print("\nFull ranking:")
        print(summary[["model", "f1_macro_pct_mean", "f1_macro_pct_std"]].to_string(index=False))

        # Use the parameter combination that won most often across outer folds.
        best_params = self._resolve_best_params(best_model_name)

        # Distance- and margin-based models need scaling; tree models do not.
        needs_scaler = {"KNN", "Approx RBF SVM"}
        use_scaler   = best_model_name in needs_scaler

        return best_model_name, best_params, use_scaler

    def _resolve_best_params(self, model_name: str) -> dict:
        # Read fold-level results and pick the most common best_params
        # across outer folds, then strip the "model__" prefix.
        fold_path = self.RESULTS_PATH.parent / "group_nested_kfold_cv_fold_results.csv"

        if not fold_path.exists():
            print("Fold results CSV not found - using empty params (model defaults).")
            return {}

        fold_df    = pd.read_csv(fold_path)
        model_rows = fold_df[fold_df["model"] == model_name]

        if model_rows.empty:
            print(f"No fold rows found for '{model_name}' - using model defaults.")
            return {}

        # Select the parameter combination that won in most outer folds.
        most_common = model_rows["best_params"].value_counts().idxmax()
        raw_params  = json.loads(most_common)

        clean_params = {
            k.removeprefix("model__"): v
            for k, v in raw_params.items()
        }

        print(f"\nBest params (most frequent across outer folds):")
        for k, v in clean_params.items():
            print(f"  {k}: {v}")

        return clean_params

    def _build_estimator(self, model_name: str, params: dict):
        registry = self._estimator_registry()
        if model_name not in registry:
            raise ValueError(
                f"Model '{model_name}' not in estimator registry. "
                f"Available: {list(registry.keys())}"
            )
        return registry[model_name](params)

    def run(self) -> None:
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {self.DATA_PATH}")

        df = pd.read_csv(self.DATA_PATH)

        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found")

        X = df.drop(columns=self.DROP_COLUMNS, errors="ignore")
        y = df[self.TARGET_COLUMN]

        if y.isna().any():
            raise ValueError("Target column contains missing values")

        model_name, best_params, use_scaler = self._load_best_result()

        estimator = self._build_estimator(model_name, best_params)

        pipeline = Pipeline([
            ("preprocess", make_preprocessor(X, use_scaler=use_scaler)),
            ("model",      estimator),
        ])

        print(f"\nTraining {model_name} on full dataset ({len(X)} rows)...")
        pipeline.fit(X, y)

        self.MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.MODEL_OUTPUT_PATH)

        print(f"\nModel saved to: {self.MODEL_OUTPUT_PATH}")
        print("DONE")



