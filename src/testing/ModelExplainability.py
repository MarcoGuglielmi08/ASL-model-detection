import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.util.paths import DATA_DIR, MODELS_DIR


# Generate global and local explanations for the final trained model.
class ModelExplainability:
    DATA_PATH = DATA_DIR / "asl_features_engineered.csv"
    MODEL_PATH = MODELS_DIR / "asl_model.pkl"
    OUTPUT_DIR = MODELS_DIR / "explainability_figures"

    OOF_PREDICTIONS_PATH = (
        MODELS_DIR
        / "error_analysis_figures"
        / "histgradientboosting_oof_predictions.csv"
    )

    TARGET_COLUMN = "label"
    DROP_COLUMNS = ["label", "filepath", "handedness"]

    RANDOM_STATE = 42
    PERMUTATION_SAMPLE_SIZE = 5000
    PERMUTATION_REPEATS = 2
    SHAP_BACKGROUND_SIZE = 1000
    TOP_N_FEATURES = 20

    PERMUTATION_CSV = OUTPUT_DIR / "figure_8_permutation_importance.csv"
    PERMUTATION_FIG = OUTPUT_DIR / "figure_8_permutation_importance.png"
    SHAP_CSV = OUTPUT_DIR / "figure_9_local_shap_values.csv"
    SHAP_METADATA_CSV = OUTPUT_DIR / "figure_9_local_shap_metadata.csv"
    SHAP_FIG = OUTPUT_DIR / "figure_9_local_shap_waterfall.png"

    def _load_inputs(self):
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {self.DATA_PATH}")
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")

        df = pd.read_csv(self.DATA_PATH)
        X = df.drop(columns=self.DROP_COLUMNS, errors="ignore")
        y = df[self.TARGET_COLUMN]
        pipeline = joblib.load(self.MODEL_PATH)
        return df, X, y, pipeline

    def _stratified_sample(self, X, y, sample_size):
        # Sample without losing class coverage, keeping class proportions stable.
        sample_size = min(sample_size, len(X))
        sample_size = max(sample_size, y.nunique())

        if sample_size >= len(X):
            return X, y

        _, X_sample, _, y_sample = train_test_split(
            X,
            y,
            test_size=sample_size,
            stratify=y,
            random_state=self.RANDOM_STATE,
        )
        return X_sample, y_sample

    def _clean_feature_names(self, names):
        cleaned = []
        for name in names:
            for prefix in ("num__", "cat__"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            cleaned.append(name)
        return cleaned

    def _get_transformed_feature_names(self, pipeline, X):
        # Recover feature names after sklearn preprocessing.
        preprocessor = pipeline.named_steps["preprocess"]
        try:
            names = preprocessor.get_feature_names_out()
            return self._clean_feature_names([str(name) for name in names])
        except Exception:
            return list(X.columns)

    def _run_permutation_importance(self, X, y, pipeline):
        # Measure how much macro F1 drops when each feature is shuffled.
        X_sample, y_sample = self._stratified_sample(
            X,
            y,
            self.PERMUTATION_SAMPLE_SIZE,
        )

        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        feature_names = self._get_transformed_feature_names(pipeline, X)

        X_transformed = preprocessor.transform(X_sample)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        def scorer(estimator, X_eval, y_eval):
            y_pred = estimator.predict(X_eval)
            return f1_score(y_eval, y_pred, average="macro", zero_division=0)

        result = permutation_importance(
            estimator,
            X_transformed,
            y_sample,
            scoring=scorer,
            n_repeats=self.PERMUTATION_REPEATS,
            random_state=self.RANDOM_STATE,
            n_jobs=1,
        )

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
        }).sort_values("importance_mean", ascending=False)

        importance_df.to_csv(self.PERMUTATION_CSV, index=False)
        self._plot_permutation_importance(importance_df)

    def _plot_permutation_importance(self, importance_df):
        top = importance_df.head(self.TOP_N_FEATURES).iloc[::-1]

        plt.figure(figsize=(10, 7))
        plt.barh(
            top["feature"],
            top["importance_mean"],
            color="#2979ff",
            alpha=0.9,
        )
        plt.xlabel("Macro F1 decrease after permutation")
        plt.ylabel("Feature")
        plt.title("Global Feature Importance - Permutation Importance")
        plt.tight_layout()
        plt.savefig(self.PERMUTATION_FIG, dpi=300)
        plt.close()

    def _select_oof_misclassification(self, df, X, y):
        # Pick a high-confidence outer-fold mistake for local explanation.
        if not self.OOF_PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"OOF predictions not found: {self.OOF_PREDICTIONS_PATH}")

        predictions = pd.read_csv(self.OOF_PREDICTIONS_PATH)
        required = {"filepath", "true_label", "predicted_label", "is_correct"}
        if not required.issubset(predictions.columns):
            raise ValueError("OOF predictions CSV is missing required columns")

        mistakes = predictions[predictions["is_correct"] == False].copy()
        if mistakes.empty:
            raise ValueError("No misclassified OOF samples found")

        if "confidence" in mistakes.columns:
            mistakes = mistakes.sort_values("confidence", ascending=False)

        filepath_to_index = pd.Series(df.index, index=df["filepath"]).to_dict()
        for _, row in mistakes.iterrows():
            chosen_index = filepath_to_index.get(row["filepath"])
            if chosen_index is not None:
                return {
                    "X": X.loc[[chosen_index]],
                    "true_label": y.loc[chosen_index],
                    "predicted_label": row["predicted_label"],
                    "reason": "outer_fold_misclassification_from_oof_predictions",
                }

        raise ValueError("Could not match OOF predictions to engineered dataset rows")

    def _run_local_shap(self, df, X, y, pipeline):
        # Compute SHAP values for the selected misclassified sample.
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is required for local explainability. "
                "Install it with: pip install shap"
            ) from exc

        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        feature_names = self._get_transformed_feature_names(pipeline, X)

        background_X, _ = self._stratified_sample(X, y, self.SHAP_BACKGROUND_SIZE)
        local = self._select_oof_misclassification(df, X, y)

        background_transformed = preprocessor.transform(background_X)
        local_transformed = preprocessor.transform(local["X"])

        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()
        if hasattr(local_transformed, "toarray"):
            local_transformed = local_transformed.toarray()

        explainer = shap.TreeExplainer(estimator, data=background_transformed)
        explanation = explainer(local_transformed)

        predicted_label = str(pipeline.predict(local["X"])[0])
        class_index = self._resolve_class_index(estimator, predicted_label)
        single = self._single_output_explanation(
            shap,
            explanation,
            local_transformed[0],
            feature_names,
            class_index,
        )

        contributions = pd.DataFrame({
            "feature": feature_names,
            "feature_value": local_transformed[0],
            "shap_value": single.values,
            "abs_shap_value": np.abs(single.values),
            "true_label": str(local["true_label"]),
            "predicted_label": predicted_label,
            "selection_reason": local["reason"],
        }).sort_values("abs_shap_value", ascending=False)
        contributions.to_csv(self.SHAP_CSV, index=False)

        pd.DataFrame([{
            "true_label": str(local["true_label"]),
            "predicted_label": predicted_label,
            "base_value": float(single.base_values),
            "final_output": float(single.base_values + np.sum(single.values)),
            "selection_reason": local["reason"],
        }]).to_csv(self.SHAP_METADATA_CSV, index=False)

        shap.plots.waterfall(single, max_display=15, show=False)
        plt.title(
            f"Local SHAP Explanation - predicted {predicted_label}, "
            f"true {local['true_label']}"
        )
        plt.tight_layout()
        plt.savefig(self.SHAP_FIG, dpi=300, bbox_inches="tight")
        plt.close()

    def _resolve_class_index(self, estimator, predicted_label):
        # Find the class column that corresponds to the predicted label.
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            return 0

        matches = np.where(classes.astype(str) == str(predicted_label))[0]
        if len(matches) == 0:
            return 0
        return int(matches[0])

    def _single_output_explanation(self, shap, explanation, data, feature_names, class_index):
        # Normalize SHAP output shapes to a single-class Explanation object.
        values = explanation.values
        base_values = explanation.base_values

        if isinstance(values, list):
            class_values = values[class_index][0]
        elif values.ndim == 3:
            if values.shape[1] == len(feature_names):
                class_values = values[0, :, class_index]
            else:
                class_values = values[0, class_index, :]
        else:
            class_values = values[0]

        if isinstance(base_values, list):
            class_base = base_values[class_index]
        elif np.ndim(base_values) == 2:
            class_base = base_values[0, class_index]
        else:
            class_base = base_values[0] if np.ndim(base_values) else base_values

        return shap.Explanation(
            values=class_values,
            base_values=class_base,
            data=data,
            feature_names=feature_names,
        )

    def run(self):
        print("Running model explainability...")

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df, X, y, pipeline = self._load_inputs()

        self._run_permutation_importance(X, y, pipeline)
        print(f"Permutation importance CSV: {self.PERMUTATION_CSV}")
        print(f"Permutation importance figure: {self.PERMUTATION_FIG}")

        self._run_local_shap(df, X, y, pipeline)
        print(f"Local SHAP CSV: {self.SHAP_CSV}")
        print(f"Local SHAP metadata CSV: {self.SHAP_METADATA_CSV}")
        print(f"Local SHAP figure: {self.SHAP_FIG}")
        print("Model explainability complete.")


if __name__ == "__main__":
    ModelExplainability().run()
