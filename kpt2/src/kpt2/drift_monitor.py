"""
Drift monitoring.

Two independent drift signals are tracked per merchant:

    1. Marking instability: merchants whose (rider_arrival - for_time)
       offset has high variance are inconsistent markers even after
       correcting for their average bias.
    2. Shadow divergence: merchants where the FOR-independent shadow
       model disagrees persistently with the bias-corrected FOR signal,
       which flags possible gaming or a structural change the mean-bias
       correction hasn't caught up with yet.
"""

from __future__ import annotations

import pandas as pd

from .config import SimulationConfig


def detect_marking_instability(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    merchant_std = df.groupby("merchant_id")["offset"].std().rename("offset_std")
    unstable = merchant_std[merchant_std > config.drift_std_threshold]
    return unstable.to_frame()


def detect_shadow_divergence(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    divergence = (df["for_adj_kpt"] - df["shadow_kpt"]).abs()
    df["shadow_divergence"] = divergence

    per_merchant = df.groupby("merchant_id")["shadow_divergence"].mean()
    z_scores = (per_merchant - per_merchant.mean()) / per_merchant.std()
    flagged = per_merchant[z_scores.abs() > z_threshold]
    return flagged.to_frame(name="mean_shadow_divergence")


def run_drift_report(df: pd.DataFrame, config: SimulationConfig) -> dict:
    unstable = detect_marking_instability(df, config)
    diverged = detect_shadow_divergence(df)

    print(f"\nMerchants showing unstable marking behavior: {len(unstable)}")
    print(f"Merchants showing shadow-model divergence:    {len(diverged)}")

    return {"unstable_merchants": unstable, "diverged_merchants": diverged}
