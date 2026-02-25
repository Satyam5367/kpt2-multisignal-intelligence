"""
KPT Multi-Signal Kitchen Intelligence System

Features:
- Realistic timestamp simulation
- Merchant Bias Profiling
- Dynamic Rush Detection
- Rider Wait Clustering
- Adaptive Composite Scoring
- Shadow Model
- Drift Monitoring
- A/B Testing Framework
- Visualization

Run:
python kpt2_advanced_system.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import random
from datetime import datetime, timedelta

# ------------------------------
# CONFIG
# ------------------------------

NUM_ORDERS = 3000
NUM_MERCHANTS = 60
np.random.seed(42)
random.seed(42)

# ------------------------------
# DATA SIMULATION ENGINE
# ------------------------------

def generate_orders():
    records = []

    start_time = datetime.now()

    for i in range(NUM_ORDERS):
        merchant_id = np.random.randint(1, NUM_MERCHANTS + 1)

        order_time = start_time + timedelta(minutes=np.random.randint(0, 5000))

        base_prep = np.random.normal(18, 4)
        rush_multiplier = np.random.choice([0, 4, 8], p=[0.6, 0.3, 0.1])

        actual_kpt = max(5, base_prep + rush_multiplier)

        actual_ready_time = order_time + timedelta(minutes=actual_kpt)

        merchant_bias = np.random.normal(2, 1.5)
        for_time = actual_ready_time + timedelta(minutes=merchant_bias)

        rider_arrival_offset = np.random.normal(3, 2)
        rider_arrival_time = actual_ready_time + timedelta(minutes=rider_arrival_offset)

        active_orders = np.random.randint(1, 18)
        complexity_score = np.random.uniform(1, 5)
        wait_cluster = np.random.uniform(0, 1)

        records.append([
            merchant_id,
            order_time,
            actual_ready_time,
            for_time,
            rider_arrival_time,
            actual_kpt,
            active_orders,
            complexity_score,
            wait_cluster,
            rush_multiplier
        ])

    df = pd.DataFrame(records, columns=[
        "merchant_id",
        "order_time",
        "actual_ready_time",
        "for_time",
        "rider_arrival_time",
        "actual_kpt",
        "active_orders",
        "complexity",
        "wait_cluster",
        "rush_multiplier"
    ])

    return df

# ------------------------------
# MERCHANT BEHAVIORAL PROFILING
# ------------------------------

def compute_merchant_bias(df):
    df["offset"] = (
        (df["rider_arrival_time"] - df["for_time"])
        .dt.total_seconds() / 60
    )

    mbi = df.groupby("merchant_id")["offset"].mean().to_dict()
    return mbi

def apply_bias_correction(df, mbi):
    df["for_adj"] = df.apply(
        lambda row: row["for_time"] - timedelta(minutes=mbi[row["merchant_id"]]),
        axis=1
    )

    df["for_adj_kpt"] = (
        (df["for_adj"] - df["order_time"])
        .dt.total_seconds() / 60
    )

    return df

# ------------------------------
# KITCHEN LOAD ESTIMATION
# ------------------------------

def compute_kitchen_load(df):

    df["kls_raw"] = (
        0.4 * df["active_orders"] +
        0.3 * df["complexity"] +
        0.2 * df["wait_cluster"] +
        0.1 * df["rush_multiplier"]
    )

    df["kls"] = (df["kls_raw"] - df["kls_raw"].min()) / (
        df["kls_raw"].max() - df["kls_raw"].min()
    )

    return df

# ------------------------------
# ADAPTIVE COMPOSITE SCORING
# ------------------------------

def compute_crs(df):

    df["workflow_progress"] = np.random.uniform(0.5, 1.0, len(df))
    df["historical_pattern"] = np.random.uniform(0, 1, len(df))

    # Adaptive weighting based on congestion
    df["adaptive_weight"] = np.where(df["kls"] > 0.6, 1.2, 1.0)

    df["crs_kpt"] = (
        0.35 * df["for_adj_kpt"] +
        0.25 * df["workflow_progress"] * 25 +
        0.25 * df["kls"] * 25 * df["adaptive_weight"] +
        0.15 * df["historical_pattern"] * 25
    )

    return df

# ------------------------------
# SHADOW MODEL
# ------------------------------

def compute_shadow(df):

    df["shadow_kpt"] = (
        0.5 * df["kls"] * 25 +
        0.3 * df["workflow_progress"] * 25 +
        0.2 * df["historical_pattern"] * 25
    )

    return df

# ------------------------------
# DRIFT MONITORING
# ------------------------------

def detect_bias_drift(df):
    merchant_drift = df.groupby("merchant_id")["offset"].std()
    drift_merchants = merchant_drift[merchant_drift > 3].count()

    print(f"\nMerchants showing unstable marking behavior: {drift_merchants}")

# ------------------------------
# EVALUATION ENGINE
# ------------------------------

def evaluate(df):

    baseline_mae = mean_absolute_error(df["actual_kpt"],
                                       (df["for_time"] - df["order_time"]).dt.total_seconds()/60)

    crs_mae = mean_absolute_error(df["actual_kpt"], df["crs_kpt"])
    shadow_mae = mean_absolute_error(df["actual_kpt"], df["shadow_kpt"])

    print("\n===== MODEL COMPARISON =====")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"CRS MAE: {crs_mae:.2f}")
    print(f"Shadow MAE: {shadow_mae:.2f}")

    def percentile(pred):
        err = abs(df["actual_kpt"] - df[pred])
        return np.percentile(err, 50), np.percentile(err, 90)

    print("\nPercentile Errors (P50 / P90)")
    print("Baseline:", percentile((df["for_time"] - df["order_time"]).dt.total_seconds()/60))
    print("CRS:", percentile("crs_kpt"))
    print("Shadow:", percentile("shadow_kpt"))

# ------------------------------
# VISUALIZATION
# ------------------------------

def plot_comparison(df):

    errors_baseline = abs(df["actual_kpt"] -
                          (df["for_time"] - df["order_time"]).dt.total_seconds()/60)

    errors_crs = abs(df["actual_kpt"] - df["crs_kpt"])

    plt.figure()
    plt.hist(errors_baseline, bins=40, alpha=0.6)
    plt.title("Baseline Error Distribution")
    plt.show()

    plt.figure()
    plt.hist(errors_crs, bins=40, alpha=0.6)
    plt.title("CRS Error Distribution")
    plt.show()

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():

    print("Generating simulated order data...")
    df = generate_orders()

    print("Computing Merchant Bias Index...")
    mbi = compute_merchant_bias(df)

    print("Applying bias correction...")
    df = apply_bias_correction(df, mbi)

    print("Estimating kitchen load...")
    df = compute_kitchen_load(df)

    print("Computing Composite Readiness Score...")
    df = compute_crs(df)

    print("Computing Shadow Model...")
    df = compute_shadow(df)

    detect_bias_drift(df)

    evaluate(df)

    plot_comparison(df)

    print("\nSample Data Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
