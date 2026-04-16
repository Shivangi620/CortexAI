import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Tuple, Dict, Any

from infra.storage import get_run_dir
from infra.logger import get_logger

log = get_logger(__name__)


class DriftDetector:
    MAX_BASELINE_FEATURES = 256
    MAX_VALUES_PER_FEATURE = 2000

    def __init__(self, run_id: str):
        self.run_id = run_id

        # ✅ FIX 1: ensure directory exists before writing
        run_dir = get_run_dir(run_id)
        os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

        self.baseline_path = os.path.join(run_dir, "data", "drift_baseline.json")

    @staticmethod
    def compute_ks_drift(baseline: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
        # ✅ FIX 2: handle None safely
        if baseline is None or current is None or len(baseline) == 0 or len(current) == 0:
            return False, 1.0

        try:
            _, p_val = ks_2samp(baseline, current)
            return p_val < 0.05, p_val
        except Exception:
            return False, 1.0  # safe fallback

    @staticmethod
    def compute_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        # ✅ FIX 3: handle None safely
        if baseline is None or current is None or len(baseline) == 0 or len(current) == 0:
            return 0.0

        try:
            breakpoints = np.percentile(baseline, np.linspace(0, 100, buckets + 1))
            breakpoints = np.unique(breakpoints)

            if len(breakpoints) < 2:
                return 0.0

            expected_percents = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
            actual_percents = np.histogram(current, bins=breakpoints)[0] / len(current)

            expected_percents = np.clip(expected_percents, 1e-6, 1.0)
            actual_percents = np.clip(actual_percents, 1e-6, 1.0)

            return float(np.sum(
                (actual_percents - expected_percents) *
                np.log(actual_percents / expected_percents)
            ))

        except Exception as e:
            log.warning(f"PSI computation failed: {e}")
            return 0.0

    def fit_baseline(self, df: pd.DataFrame):
        """Calculates and stores standard baseline distribution constraints."""

        # ✅ FIX 4: handle empty or None dataframe
        if df is None or df.empty:
            log.warning("Baseline fit skipped: empty dataframe")
            return

        baseline = {}

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns[: self.MAX_BASELINE_FEATURES])
        for col in numeric_cols:
            try:
                values = df[col].dropna().to_numpy()
                if len(values) > self.MAX_VALUES_PER_FEATURE:
                    sample_idx = np.linspace(0, len(values) - 1, self.MAX_VALUES_PER_FEATURE, dtype=int)
                    values = values[sample_idx]
                baseline[col] = values.tolist()
            except Exception:
                continue  # skip bad columns safely

        # ✅ FIX 5: safe file write
        try:
            with open(self.baseline_path, "w") as f:
                json.dump(baseline, f)
        except Exception as e:
            log.error(f"Failed to save baseline: {e}")
            return

        log.info(f"Drift Baseline fitted and saved for {len(baseline)} numeric features.")

    def compare_drift(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """Compares target dataframe against fitted bounds. Returns score and alerts."""

        # ✅ FIX 6: validate input dataframe
        if new_df is None or new_df.empty:
            return {
                "error": "Empty input data",
                "drift_score_pct": 0,
                "alerts": []
            }

        if not os.path.exists(self.baseline_path):
            return {
                "error": "No baseline fitted for this model.",
                "drift_score_pct": 0,
                "alerts": []
            }

        # ✅ FIX 7: safe file read
        try:
            with open(self.baseline_path, "r") as f:
                baseline_data = json.load(f)
        except Exception as e:
            log.error(f"Failed to load baseline: {e}")
            return {
                "error": "Corrupted baseline data",
                "drift_score_pct": 0,
                "alerts": []
            }

        drifted_features = 0
        total_eval = 0
        alerts = []

        for col, base_values in baseline_data.items():
            if col not in new_df.columns:
                continue

            try:
                current_vals = new_df[col].dropna().values
                base_vals = np.array(base_values)
            except Exception:
                continue  # skip invalid columns

            total_eval += 1

            is_ks_drift, p_val = self.compute_ks_drift(base_vals, current_vals)
            psi_val = self.compute_psi(base_vals, current_vals)

            if is_ks_drift or psi_val >= 0.2:
                drifted_features += 1
                alerts.append(
                    f"Feature '{col}': PSI={psi_val:.2f}, KS p={p_val:.3f}. Significant Drift."
                )

        # ✅ FIX 8: safe division
        score = (drifted_features / total_eval * 100) if total_eval > 0 else 0

        if score > 20:
            log.warning(f"Major Data Drift Detected! Score: {score:.1f}%")

        return {
            "drift_score_pct": score,
            "drifted_features": drifted_features,
            "total_evaluated": total_eval,
            "alerts": alerts
        }
