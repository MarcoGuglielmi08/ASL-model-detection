import json
import os
from time import perf_counter

from src.util.preprocessor import make_preprocessor
from src.util.paths import DATA_DIR, MODELS_DIR
from util.paths import NESTED_DIR, RESULTS_DIR

# Avoid CPU oversubscription when GridSearchCV runs multiple fits in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# Run grouped nested cross-validation and store report-ready outputs.
class ModelNestedKFold:
    DATA_PATH = DATA_DIR / "asl_features_engineered.csv"
    SUMMARY_OUTPUT_PATH = NESTED_DIR/ "group_nested_kfold_cv_results.csv"
    FOLD_OUTPUT_PATH = NESTED_DIR / "group_nested_kfold_cv_fold_results.csv"
    CV_RESULTS_DIR = RESULTS_DIR / "grid_search_cv_results"
    ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
    OOF_PREDICTIONS_PATH = ERROR_ANALYSIS_DIR / "histgradientboosting_oof_predictions.csv"

    TARGET_COLUMN = "label"
    DROP_COLUMNS  = ["label", "filepath", "handedness"]

    OUTER_FOLDS  = 5
    INNER_FOLDS  = 4
    RANDOM_STATE = 42

    CPU_COUNT       = os.cpu_count() or 1
    CV_N_JOBS       = int(os.environ.get("ASL_CV_N_JOBS", str(CPU_COUNT)))
    CV_PRE_DISPATCH = os.environ.get("ASL_CV_PRE_DISPATCH", "2*n_jobs")
    CV_VERBOSE      = 2

    MODELS_TO_RUN = [
        "KNN",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
        "Approx RBF SVM",
    ]

    def build_groups(self, X: pd.DataFrame) -> pd.Series:
        # Group duplicate feature rows so identical samples stay in the same fold.
        return hash_pandas_object(X, index=False)

    def make_stratified_group_splits(
        self,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int,
        random_state: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Create folds that preserve labels while keeping groups together.

        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        y_values     = pd.Series(y).reset_index(drop=True)
        group_values = pd.Series(groups).reset_index(drop=True)

        group_codes, _ = pd.factorize(group_values, sort=False)
        class_codes, _ = pd.factorize(y_values, sort=True)

        n_groups  = int(group_codes.max() + 1)
        n_classes = int(class_codes.max() + 1)

        if n_groups < n_splits:
            raise ValueError("Number of unique groups must be >= n_splits")

        group_class_counts = np.zeros((n_groups, n_classes), dtype=np.float64)
        np.add.at(group_class_counts, (group_codes, class_codes), 1.0)

        class_totals        = group_class_counts.sum(axis=0)
        target_class_counts = class_totals / float(n_splits)
        class_scale         = np.maximum(target_class_counts, 1.0)
        target_fold_size    = len(y_values) / float(n_splits)

        fold_class_counts = np.zeros((n_splits, n_classes), dtype=np.float64)
        fold_sizes        = np.zeros(n_splits, dtype=np.float64)
        group_fold        = np.full(n_groups, fill_value=-1, dtype=np.int32)
        group_sizes       = group_class_counts.sum(axis=1)

        rng         = np.random.default_rng(random_state)
        group_order = rng.permutation(n_groups)
        # Assign larger groups first to improve fold-size and class balance.
        group_order = group_order[np.argsort(-group_sizes[group_order], kind="mergesort")]

        for group_idx in group_order:
            group_counts = group_class_counts[group_idx]
            group_size   = group_sizes[group_idx]

            candidate_class_counts = fold_class_counts + group_counts

            current_class_errors   = ((fold_class_counts - target_class_counts) / class_scale) ** 2
            candidate_class_errors = ((candidate_class_counts - target_class_counts) / class_scale) ** 2
            class_scores           = np.sum(candidate_class_errors - current_class_errors, axis=1)

            candidate_fold_sizes  = fold_sizes + group_size
            current_size_errors   = ((fold_sizes - target_fold_size) / target_fold_size) ** 2
            candidate_size_errors = ((candidate_fold_sizes - target_fold_size) / target_fold_size) ** 2
            size_scores           = candidate_size_errors - current_size_errors

            scores    = class_scores + size_scores + fold_sizes * 1e-12
            best_fold = int(np.argmin(scores))

            group_fold[group_idx]          = best_fold
            fold_class_counts[best_fold]  += group_counts
            fold_sizes[best_fold]         += group_size

        sample_fold  = group_fold[group_codes]
        all_indices  = np.arange(len(y_values))

        splits = []
        for fold_idx in range(n_splits):
            test_mask = sample_fold == fold_idx
            if not np.any(test_mask):
                raise ValueError(f"Fold {fold_idx + 1} has no test samples")
            splits.append((all_indices[~test_mask], all_indices[test_mask]))

        return splits

    def make_model_specs(self) -> dict:
        # Return estimators, scaler requirements, and grids for each candidate.
        return {
            "KNN": {
                "use_scaler": True,
                "estimator": KNeighborsClassifier(),
                "param_grid": {
                    "model__n_neighbors": [3, 5, 7, 9, 11],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
            },

            "Decision Tree": {
                "use_scaler": False,
                "estimator": DecisionTreeClassifier(random_state=self.RANDOM_STATE),
                "param_grid": {
                    "model__criterion": ["gini", "entropy"],
                    "model__max_depth": [None, 20, 30, 40],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 5],
                },
            },

            "Random Forest": {
                "use_scaler": False,
                "estimator": RandomForestClassifier(
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=self.RANDOM_STATE,
                    n_jobs=1,
                ),
                "param_grid": {
                    "model__n_estimators": [500, 800, 1000],
                    "model__max_depth": [20, 30, None],
                    "model__max_features": ["log2", "sqrt"],
                    "model__min_samples_leaf": [1, 2],
                },
            },

            "HistGradientBoosting": {
                "use_scaler": False,
                "estimator": HistGradientBoostingClassifier(
                    l2_regularization=0.0,
                    early_stopping=False,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=self.RANDOM_STATE,
                ),
                "param_grid": [
                    {
                        "model__learning_rate": [0.05],
                        "model__max_iter": [800],
                        "model__max_leaf_nodes": [31],
                        "model__min_samples_leaf": [40],
                        "model__l2_regularization": [0.0],
                        "model__early_stopping": [False],
                    },
                    {
                        "model__learning_rate": [0.05],
                        "model__max_iter": [400],
                        "model__max_leaf_nodes": [31, 63],
                        "model__min_samples_leaf": [20, 40],
                        "model__l2_regularization": [0.0, 0.1, 1.0],
                        "model__early_stopping": [True],
                        "model__n_iter_no_change": [20],
                    },
                    {
                        "model__learning_rate": [0.03],
                        "model__max_iter": [600, 800],
                        "model__max_leaf_nodes": [31],
                        "model__min_samples_leaf": [20],
                        "model__l2_regularization": [0.0],
                        "model__early_stopping": [True],
                        "model__n_iter_no_change": [20],
                    },
                    {
                        "model__learning_rate": [0.07],
                        "model__max_iter": [300],
                        "model__max_leaf_nodes": [31],
                        "model__min_samples_leaf": [20],
                        "model__l2_regularization": [0.0],
                        "model__early_stopping": [True],
                        "model__n_iter_no_change": [20],
                    },
                    {
                        "model__learning_rate": [0.03],
                        "model__max_iter": [800],
                        "model__max_leaf_nodes": [63],
                        "model__min_samples_leaf": [10],
                        "model__l2_regularization": [0.1],
                        "model__early_stopping": [True],
                        "model__n_iter_no_change": [30],
                    },
                    {
                        "model__learning_rate": [0.05],
                        "model__max_iter": [400],
                        "model__max_leaf_nodes": [127],
                        "model__min_samples_leaf": [20],
                        "model__l2_regularization": [0.0],
                        "model__early_stopping": [True],
                        "model__n_iter_no_change": [20],
                    },
                ],
            },

            "Approx RBF SVM": {
                "use_scaler": True,
                "estimator": Pipeline([
                    ("rbf_features", Nystroem(kernel="rbf", random_state=self.RANDOM_STATE)),
                    ("linear_svm", LinearSVC(class_weight="balanced", max_iter=10000, random_state=self.RANDOM_STATE)),
                ]),
                "param_grid": {
                    "model__rbf_features__n_components": [300, 500],
                    "model__rbf_features__gamma": [0.0005, 0.001, 0.003, 0.01],
                    "model__linear_svm__C": [0.5, 1.0, 5.0, 10.0],
                },
            },
        }

    def filter_model_specs(self, model_specs: dict, models_to_run: list[str]) -> dict:
        unknown = [m for m in models_to_run if m not in model_specs]
        if unknown:
            raise ValueError("Unknown models in MODELS_TO_RUN: " + ", ".join(unknown))
        return {m: model_specs[m] for m in models_to_run}

    def compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        # Compute the metrics reported for each outer fold.
        return {
            "accuracy_pct":        float(accuracy_score(y_true, y_pred) * 100.0),
            "precision_macro_pct": float(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
            "recall_macro_pct":    float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
            "f1_macro_pct":        float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
        }

    def build_summary(self, fold_results: pd.DataFrame) -> pd.DataFrame:
        # Aggregate outer-fold results into the model ranking CSV.
        metric_columns = [
            "accuracy_pct", "precision_macro_pct",
            "recall_macro_pct", "f1_macro_pct", "inner_best_f1_macro_pct",
        ]
        rows = []
        for model_name, model_rows in fold_results.groupby("model"):
            row = {"model": model_name, "outer_folds": int(len(model_rows))}
            for col in metric_columns:
                row[f"{col}_mean"] = float(model_rows[col].mean())
                row[f"{col}_std"]  = float(model_rows[col].std(ddof=0))
            row["best_params_by_fold"] = " | ".join(model_rows["best_params"].tolist())
            rows.append(row)
        return pd.DataFrame(rows).sort_values(by="f1_macro_pct_mean", ascending=False)

    def save_results(self, fold_rows: list[dict]) -> None:
        # Persist fold-level results and the aggregated summary.
        if not fold_rows:
            return
        fold_results = pd.DataFrame(fold_rows)
        summary      = self.build_summary(fold_results)
        self.SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fold_results.to_csv(self.FOLD_OUTPUT_PATH, index=False)
        summary.to_csv(self.SUMMARY_OUTPUT_PATH, index=False)

    def model_slug(self, model_name: str) -> str:
        return model_name.lower().replace(" ", "_")

    def save_cv_results(self, outer_fold: int, model_name: str, cv_results: dict) -> None:
        # Save GridSearchCV rows for later inspection and report figures.
        self.CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame(cv_results)
        results.insert(0, "outer_fold", outer_fold)
        results.insert(1, "model", model_name)

        for column in results.columns:
            if column.startswith("param_"):
                results[column] = results[column].astype(str)

        output_path = self.CV_RESULTS_DIR / f"{self.model_slug(model_name)}_outer_fold_{outer_fold}.csv"
        results.to_csv(output_path, index=False)

    def save_oof_predictions(self, prediction_rows: list[pd.DataFrame]) -> None:
        # Save per-sample outer-fold predictions for error analysis figures.
        if not prediction_rows:
            return
        self.ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        predictions.to_csv(self.OOF_PREDICTIONS_PATH, index=False)

    def run(self) -> None:
        if self.OUTER_FOLDS < 2:
            raise ValueError("OUTER_FOLDS must be >= 2")
        if self.INNER_FOLDS < 2:
            raise ValueError("INNER_FOLDS must be >= 2")
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {self.DATA_PATH}")

        df = pd.read_csv(self.DATA_PATH)
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found")

        X = df.drop(columns=self.DROP_COLUMNS, errors="ignore")
        y = df[self.TARGET_COLUMN]

        if y.isna().any():
            raise ValueError("Target column contains missing values")

        groups = self.build_groups(X)

        print(f"Rows: {len(df)}")
        print(f"Unique groups: {groups.nunique()}")
        print(f"Outer folds: {self.OUTER_FOLDS} | Inner folds: {self.INNER_FOLDS}")
        print(f"CPU cores: {self.CPU_COUNT} | GridSearchCV n_jobs: {self.CV_N_JOBS}")
        print("Models to run: " + ", ".join(self.MODELS_TO_RUN))

        print("\nBuilding outer splits...")
        t = perf_counter()
        outer_splits = self.make_stratified_group_splits(y, groups, self.OUTER_FOLDS, self.RANDOM_STATE)
        print(f"Outer splits ready in {perf_counter() - t:.2f}s")

        model_specs        = self.filter_model_specs(self.make_model_specs(), self.MODELS_TO_RUN)
        total_combinations = sum(len(list(ParameterGrid(s["param_grid"]))) for s in model_specs.values())

        print(f"Total parameter combinations per outer fold: {total_combinations}")
        print(f"Total inner fits across all outer folds: {total_combinations * self.INNER_FOLDS * self.OUTER_FOLDS}")

        fold_rows = []
        oof_prediction_rows = []

        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            print(f"\nOuter fold {outer_fold}/{self.OUTER_FOLDS}")

            X_train      = X.iloc[train_idx]
            X_test       = X.iloc[test_idx]
            y_train      = y.iloc[train_idx]
            y_test       = y.iloc[test_idx]
            groups_train = groups.iloc[train_idx]

            print("Building inner splits...")
            t = perf_counter()
            inner_splits = self.make_stratified_group_splits(
                y_train, groups_train, self.INNER_FOLDS, self.RANDOM_STATE + outer_fold
            )
            print(f"Inner splits ready in {perf_counter() - t:.2f}s")

            for model_name, spec in model_specs.items():
                print(f"\nTuning: {model_name}")

                n_combos = len(list(ParameterGrid(spec["param_grid"])))
                print(f"Inner fits: {n_combos} combinations x {self.INNER_FOLDS} folds = {n_combos * self.INNER_FOLDS}")

                pipeline = Pipeline([
                    ("preprocess", make_preprocessor(X_train, use_scaler=spec["use_scaler"])),
                    ("model", spec["estimator"]),
                ])

                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=spec["param_grid"],
                    scoring="f1_macro",
                    cv=inner_splits,
                    n_jobs=self.CV_N_JOBS,
                    pre_dispatch=self.CV_PRE_DISPATCH,
                    verbose=self.CV_VERBOSE,
                    refit=True,
                    error_score="raise",
                )

                t = perf_counter()
                search.fit(X_train, y_train)
                print(f"Tuning completed in {perf_counter() - t:.2f}s")
                self.save_cv_results(outer_fold, model_name, search.cv_results_)

                y_pred      = search.predict(X_test)
                metrics     = self.compute_metrics(y_test, y_pred)
                best_params = json.dumps(search.best_params_, sort_keys=True)

                if model_name == "HistGradientBoosting":
                    confidence = pd.NA
                    if hasattr(search.best_estimator_, "predict_proba"):
                        probabilities = search.predict_proba(X_test)
                        confidence = probabilities.max(axis=1)

                    fold_predictions = pd.DataFrame({
                        "outer_fold": outer_fold,
                        "filepath": df.iloc[test_idx]["filepath"].to_numpy()
                            if "filepath" in df.columns else pd.NA,
                        "true_label": y_test.to_numpy(),
                        "predicted_label": y_pred,
                        "confidence": confidence,
                    })
                    fold_predictions["is_correct"] = (
                        fold_predictions["true_label"] == fold_predictions["predicted_label"]
                    )
                    oof_prediction_rows.append(fold_predictions)
                    self.save_oof_predictions(oof_prediction_rows)

                fold_rows.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "inner_best_f1_macro_pct": float(search.best_score_ * 100.0),
                    "best_params": best_params,
                    **metrics,
                })

                print(f"Best inner F1 Macro: {search.best_score_ * 100.0:.2f}%")
                print(f"Outer F1 Macro: {metrics['f1_macro_pct']:.2f}%")
                print(f"Best params: {best_params}")

                self.save_results(fold_rows)

        self.save_results(fold_rows)
        self.save_oof_predictions(oof_prediction_rows)
        print(f"\nFold results:    {self.FOLD_OUTPUT_PATH}")
        print(f"Summary results: {self.SUMMARY_OUTPUT_PATH}")
        print(f"OOF predictions: {self.OOF_PREDICTIONS_PATH}")
        print("Nested cross-validation complete.")
