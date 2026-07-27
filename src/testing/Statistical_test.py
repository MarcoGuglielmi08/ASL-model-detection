from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon


# Run non-parametric tests on the outer-fold model scores.
class StatisticalTesting:
    CSV_PATH = Path("models/group_nested_kfold_cv_fold_results.csv")
    OUTPUT_DIR = Path("statistical_tests")
    SCORE_COL = "f1_macro_pct"
    BEST_MODEL = "HistGradientBoosting"
    ALPHA = 0.05

    def run(self) -> None:
        print("Running statistical tests...")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.CSV_PATH)
        pivot = df.pivot_table(
            index="outer_fold",
            columns="model",
            values=self.SCORE_COL,
        ).sort_index()

        wilcoxon_df = self._run_wilcoxon(df)
        friedman_stat, friedman_p = self._run_friedman(pivot)
        nemenyi_df = self._run_nemenyi(pivot) if friedman_p < self.ALPHA else None

        self._save_results(wilcoxon_df, friedman_stat, friedman_p, nemenyi_df)
        print("Statistical testing complete.")

    def _run_wilcoxon(self, df: pd.DataFrame) -> pd.DataFrame:
        # Compare the selected model against each alternative model.
        best_scores = self._scores_for_model(df, self.BEST_MODEL)
        rows = []

        for model in sorted(df["model"].unique()):
            if model == self.BEST_MODEL:
                continue

            stat, p_value = wilcoxon(
                best_scores,
                self._scores_for_model(df, model),
                alternative="greater",
            )
            rows.append({
                "comparison": f"{self.BEST_MODEL} vs {model}",
                "W_statistic": stat,
                "p_value": round(p_value, 6),
                f"significant_{self.ALPHA}": p_value < self.ALPHA,
            })

        min_p = self._wilcoxon_min_p(len(best_scores))
        print(f"Wilcoxon done. Minimum one-sided p-value with 5 folds: {min_p:.4f}.")
        return pd.DataFrame(rows)

    def _run_friedman(self, pivot: pd.DataFrame) -> tuple[float, float]:
        # Test whether all candidate models have equivalent performance.
        stat, p_value = friedmanchisquare(
            *[pivot[model].values for model in pivot.columns]
        )
        print(f"Friedman done. statistic={stat:.6f}, p={p_value:.6f}")
        return stat, p_value

    def _run_nemenyi(self, pivot: pd.DataFrame) -> pd.DataFrame:
        # Run the post-hoc pairwise test after a significant Friedman result.
        nemenyi_df = sp.posthoc_nemenyi_friedman(pivot.values)
        nemenyi_df.index = pivot.columns
        nemenyi_df.columns = pivot.columns

        print("Nemenyi done.")
        return nemenyi_df

    def _save_results(
        self,
        wilcoxon_df: pd.DataFrame,
        friedman_stat: float,
        friedman_p: float,
        nemenyi_df: pd.DataFrame | None,
    ) -> None:
        wilcoxon_df.to_csv(self.OUTPUT_DIR / "wilcoxon_results.csv", index=False)

        pd.DataFrame([{
            "test": "Friedman",
            "statistic": round(friedman_stat, 6),
            "p_value": round(friedman_p, 6),
        }]).to_csv(self.OUTPUT_DIR / "friedman_results.csv", index=False)

        if nemenyi_df is not None:
            nemenyi_df.round(6).to_csv(self.OUTPUT_DIR / "nemenyi_results.csv")

        print(f"Results saved to {self.OUTPUT_DIR}/")

    def _scores_for_model(self, df: pd.DataFrame, model: str) -> np.ndarray:
        # Return one score per outer fold for the requested model.
        return (
            df[df["model"] == model]
            .sort_values("outer_fold")[self.SCORE_COL]
            .values
        )

    @staticmethod
    def _wilcoxon_min_p(n_pairs: int) -> float:
        # Compute the smallest one-sided Wilcoxon p-value possible for n pairs.
        a = np.arange(1, n_pairs + 1, dtype=float)
        b = np.zeros(n_pairs, dtype=float)
        _, p_value = wilcoxon(a, b, alternative="greater")
        return p_value


if __name__ == "__main__":
    StatisticalTesting().run()
