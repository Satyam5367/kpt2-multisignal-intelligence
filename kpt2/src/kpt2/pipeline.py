"""
End-to-end pipeline orchestration: simulate -> bias-correct -> load-score
-> historical pattern -> CRS -> shadow -> drift -> evaluate.
"""

from __future__ import annotations

import pandas as pd

from .composite_scoring import compute_crs
from .config import SimulationConfig
from .drift_monitor import run_drift_report
from .historical_tracker import compute_historical_pattern
from .kitchen_load import compute_kitchen_load
from .merchant_bias import apply_bias_correction, compute_merchant_bias_index
from .shadow_model import compute_shadow
from .simulation import generate_orders


def run_pipeline(config: SimulationConfig | None = None) -> dict:
    config = config or SimulationConfig()

    print("Simulating order lifecycles (heap-based congestion, sliding-window rider clustering)...")
    df = generate_orders(config)

    print("Computing Merchant Bias Index...")
    mbi = compute_merchant_bias_index(df)

    print("Applying bias correction...")
    df = apply_bias_correction(df, mbi)

    print("Estimating Kitchen Load Score...")
    df = compute_kitchen_load(df)

    print("Computing per-merchant historical pattern (rolling window)...")
    df = compute_historical_pattern(df, config)

    print("Resolving kitchen workflow dependency graphs + fitting Adaptive CRS...")
    df, crs_metadata = compute_crs(df, config)

    print("Fitting FOR-independent Shadow Model...")
    df, shadow_metadata = compute_shadow(df)

    df["baseline_kpt"] = (df["for_time"] - df["order_time"]).dt.total_seconds() / 60.0

    drift_report = run_drift_report(df, config)

    return {
        "df": df,
        "mbi": mbi,
        "crs_metadata": crs_metadata,
        "shadow_metadata": shadow_metadata,
        "drift_report": drift_report,
        "config": config,
    }
