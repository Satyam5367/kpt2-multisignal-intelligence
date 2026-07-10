"""
Adaptive Composite Readiness Score (CRS).

Rather than hand-picking fixed blend weights (the failure mode of the
original prototype, where uncorrelated random features were mixed in
with fixed coefficients and made predictions *worse*), CRS is a fitted
linear model over four signals that each carry independent, causally
grounded information:

    for_adj_kpt                 -- bias-corrected merchant FOR signal
    workflow_duration_estimate  -- graph-based critical-path estimate
    kls                         -- kitchen load / congestion score
    historical_pattern          -- per-merchant rolling average

"Adaptive" weighting is implemented as an explicit congestion-regime
interaction term (kls_high * kls): the model is allowed a different
slope on kls once congestion crosses a threshold, which is exactly the
scenario (rush hour) where merchant marking bias is least reliable and
the kitchen-load / workflow signals should be trusted more. This is
standard interaction-term feature engineering for a linear model, fit
with scikit-learn.

The model is fit on a time-ordered train split and evaluated on a
held-out time-ordered test split, so reported metrics reflect
out-of-sample performance rather than in-sample fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import SimulationConfig
from .workflow_graph import estimate_workflow_duration

CRS_FEATURE_COLUMNS = [
    "for_adj_kpt",
    "workflow_duration_estimate",
    "kls",
    "kls_high_interaction",
    "historical_pattern",
]

CONGESTION_THRESHOLD = 0.6


def add_workflow_duration_estimate(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    df["workflow_duration_estimate"] = [
        estimate_workflow_duration(
            complexity=row["complexity"],
            active_orders=int(row["active_orders"]),
            congestion_scale=config.congestion_scale_per_active_order,
            complexity_scale_divisor=config.complexity_scale_divisor,
            estimation_noise_std=config.workflow_estimate_noise_std,
            noise_seed=config.random_seed * 1_000_003 + idx,
        )
        for idx, row in df.reset_index(drop=True).iterrows()
    ]
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["kls_high"] = (df["kls"] > CONGESTION_THRESHOLD).astype(float)
    df["kls_high_interaction"] = df["kls_high"] * df["kls"]
    return df


def time_ordered_split(df: pd.DataFrame, train_frac: float = 0.7):
    ordered = df.sort_values("order_time")
    split_at = int(len(ordered) * train_frac)
    train_idx = ordered.index[:split_at]
    test_idx = ordered.index[split_at:]
    return train_idx, test_idx


def compute_crs(df: pd.DataFrame, config: SimulationConfig) -> tuple[pd.DataFrame, dict]:
    """Fit CRS on a time-ordered train split, predict on all rows.

    Returns the augmented dataframe plus a small metadata dict (learned
    coefficients, which split was used) for transparency/debugging.
    """
    df = add_workflow_duration_estimate(df, config)
    df = _add_interaction_features(df)

    train_idx, test_idx = time_ordered_split(df)

    model = LinearRegression()
    model.fit(df.loc[train_idx, CRS_FEATURE_COLUMNS], df.loc[train_idx, "actual_kpt"])

    df["crs_kpt"] = model.predict(df[CRS_FEATURE_COLUMNS])
    df["crs_kpt"] = df["crs_kpt"].clip(lower=3.0)

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    metadata = {
        "coefficients": dict(zip(CRS_FEATURE_COLUMNS, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }
    return df, metadata
