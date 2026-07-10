"""
Headless plotting (matplotlib Agg backend) -- saves PNGs to disk instead
of calling plt.show(), so the pipeline runs the same way in a CI job,
a container, or a notebook.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_error_distributions(df: pd.DataFrame, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    errors = {
        "Baseline (raw FOR)": (df["actual_kpt"] - df["baseline_kpt"]).abs(),
        "CRS": (df["actual_kpt"] - df["crs_kpt"]).abs(),
        "Shadow": (df["actual_kpt"] - df["shadow_kpt"]).abs(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, err) in zip(axes, errors.items()):
        ax.hist(err, bins=40, alpha=0.75, color="#3b6fa0")
        ax.set_title(f"{name}\nMAE={err.mean():.2f}")
        ax.set_xlabel("Absolute error (minutes)")
    axes[0].set_ylabel("Order count")
    fig.suptitle("Prep-time prediction error distributions")
    fig.tight_layout()

    path = os.path.join(output_dir, "error_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved


def plot_merchant_bias_heatmap(mbi: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    merchants = sorted(mbi.keys())
    biases = [mbi[m] for m in merchants]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(merchants, biases, color="#c0785a")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Merchant ID")
    ax.set_ylabel("Mean marking bias (minutes)")
    ax.set_title("Merchant Bias Index (MBI)")
    fig.tight_layout()

    path = os.path.join(output_dir, "merchant_bias_index.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
