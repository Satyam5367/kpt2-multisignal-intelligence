"""
Kitchen Load Score (KLS): a normalized [0, 1] congestion estimate built
from three causally-relevant, independently-derived signals:

    - active_orders   : heap-based concurrency estimate (simulation.py)
    - complexity      : per-order dish complexity
    - wait_cluster    : sliding-window rider pile-up density (simulation.py)

Unlike the original prototype (which mixed in a random rush_multiplier
term), every input here is either an observed order attribute or a
derived signal with a real causal link to prep time, so KLS is a
genuine congestion proxy rather than noise dressed up as a feature.
"""

from __future__ import annotations

import pandas as pd


def compute_kitchen_load(df: pd.DataFrame) -> pd.DataFrame:
    raw = 0.5 * df["active_orders"] + 0.3 * df["complexity"] + 0.2 * df["wait_cluster"]
    df["kls_raw"] = raw

    denom = raw.max() - raw.min()
    df["kls"] = (raw - raw.min()) / denom if denom > 0 else 0.0
    return df
