"""
Shadow KPT Model.

A FOR-independent estimate of prep time, deliberately excluding the
merchant-marked signal entirely. Its purpose is not to beat CRS (it
gives up real information by design) but to act as an independent
cross-check: if a merchant's FOR-based signal and the shadow estimate
diverge persistently, that is itself evidence of drift or gaming in the
merchant's marking behavior (see drift_monitor.py).

Fit the same way as CRS (time-ordered train/test split, linear
regression) but restricted to non-FOR features.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from .composite_scoring import time_ordered_split

SHADOW_FEATURE_COLUMNS = [
    "workflow_duration_estimate",
    "kls",
    "historical_pattern",
]


def compute_shadow(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_idx, test_idx = time_ordered_split(df)

    model = LinearRegression()
    model.fit(df.loc[train_idx, SHADOW_FEATURE_COLUMNS], df.loc[train_idx, "actual_kpt"])

    df["shadow_kpt"] = model.predict(df[SHADOW_FEATURE_COLUMNS]).clip(min=3.0)

    metadata = {
        "coefficients": dict(zip(SHADOW_FEATURE_COLUMNS, model.coef_.tolist())),
        "intercept": float(model.intercept_),
    }
    return df, metadata
