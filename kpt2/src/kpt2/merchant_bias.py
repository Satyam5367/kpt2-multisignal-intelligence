"""
Merchant Bias Index (MBI): behavioral bias profiling and correction.

Each merchant tends to mark "Food Order Ready" (FOR) systematically
early or late relative to when a rider actually arrives to collect the
order. MBI estimates that systematic offset per merchant (a simple hash
map of merchant_id -> mean offset) and subtracts it from the raw FOR
timestamp.

By construction (see simulation.py) this correction removes the FIXED
component of merchant bias but leaves the congestion-correlated
component untouched -- that residual is exactly what the Composite
Readiness Score's kitchen-load signal is designed to pick up.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def compute_merchant_bias_index(df: pd.DataFrame) -> dict[int, float]:
    """Compute each merchant's mean (rider_arrival - for_time) offset,
    in minutes. Positive offset means the merchant marks orders ready
    *before* the rider actually shows up (over-eager marking).
    """
    offset_minutes = (
        df["rider_arrival_time"] - df["for_time"]
    ).dt.total_seconds() / 60.0
    df["offset"] = offset_minutes
    return df.groupby("merchant_id")["offset"].mean().to_dict()


def apply_bias_correction(df: pd.DataFrame, mbi: dict[int, float]) -> pd.DataFrame:
    """Subtract each merchant's mean bias from their raw FOR timestamp,
    producing a bias-corrected ready time and the corresponding
    corrected KPT estimate (minutes from order placed to corrected FOR).
    """
    bias_minutes = df["merchant_id"].map(mbi)
    df["for_adj"] = df["for_time"] - bias_minutes.apply(lambda m: timedelta(minutes=m))
    df["for_adj_kpt"] = (df["for_adj"] - df["order_time"]).dt.total_seconds() / 60.0
    return df
