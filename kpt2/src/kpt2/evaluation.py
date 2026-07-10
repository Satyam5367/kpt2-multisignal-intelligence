"""
Model comparison / evaluation engine.

Reports MAE and P50/P90 absolute error for three candidate signals
against ground-truth `actual_kpt`:

    baseline_kpt -- raw, uncorrected merchant FOR signal
    crs_kpt      -- Adaptive Composite Readiness Score
    shadow_kpt   -- FOR-independent shadow model

All metrics are additionally reported on the held-out TEST split only,
since train-split performance is not a meaningful claim of predictive
quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def _percentile_errors(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    err = (actual - predicted).abs()
    return float(np.percentile(err, 50)), float(np.percentile(err, 90))


def evaluate(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Compute MAE + P50/P90 for baseline / CRS / shadow.

    If `split` is "test", restricts to df["split"] == "test" (true
    out-of-sample evaluation). If None, evaluates on the full dataset.
    """
    subset = df if split is None else df[df["split"] == split]

    rows = []
    for name, col in [
        ("Baseline (raw FOR)", "baseline_kpt"),
        ("CRS", "crs_kpt"),
        ("Shadow", "shadow_kpt"),
    ]:
        mae = mean_absolute_error(subset["actual_kpt"], subset[col])
        p50, p90 = _percentile_errors(subset["actual_kpt"], subset[col])
        rows.append({"model": name, "mae": mae, "p50_error": p50, "p90_error": p90})

    return pd.DataFrame(rows)


def print_evaluation_report(df: pd.DataFrame) -> None:
    print("\n===== MODEL COMPARISON (full dataset) =====")
    print(evaluate(df).to_string(index=False))

    print("\n===== MODEL COMPARISON (held-out test split only) =====")
    print(evaluate(df, split="test").to_string(index=False))
