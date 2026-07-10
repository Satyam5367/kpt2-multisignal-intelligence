"""
KPT2 - Multi-Signal Kitchen Intelligence System
CLI entry point.

Usage:
    python main.py
    python main.py --num-orders 5000 --num-merchants 80
    python main.py --store-db --db-url "mysql+pymysql://user:pass@localhost/kpt2"
    python main.py --store-db --db-url "sqlite:///kpt2_local.db"
"""

from __future__ import annotations

import argparse

from kpt2.config import SimulationConfig
from kpt2.evaluation import print_evaluation_report
from kpt2.pipeline import run_pipeline
from kpt2.visualization import plot_error_distributions, plot_merchant_bias_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KPT2 pipeline.")
    parser.add_argument("--num-orders", type=int, default=3000)
    parser.add_argument("--num-merchants", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory to save plots."
    )
    parser.add_argument(
        "--store-db", action="store_true", help="Persist results to a database."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="sqlite:///kpt2_local.db",
        help="SQLAlchemy connection URL (MySQL in production, SQLite for local use).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SimulationConfig(
        num_orders=args.num_orders,
        num_merchants=args.num_merchants,
        random_seed=args.seed,
    )

    result = run_pipeline(config)
    df = result["df"]

    print_evaluation_report(df)

    print(f"\nSaving plots to {args.output_dir}/ ...")
    plot_error_distributions(df, args.output_dir)
    plot_merchant_bias_heatmap(result["mbi"], args.output_dir)

    if args.store_db:
        from kpt2.db.db_manager import store_pipeline_results, get_engine

        print(f"\nPersisting results to {args.db_url} ...")
        engine = get_engine(args.db_url)
        store_pipeline_results(engine, df, result["mbi"])
        print("Done.")

    print("\nSample Data Preview:")
    print(
        df[
            [
                "merchant_id",
                "actual_kpt",
                "baseline_kpt",
                "crs_kpt",
                "shadow_kpt",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
