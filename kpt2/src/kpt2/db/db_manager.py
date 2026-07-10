"""
Database access layer.

`schema.sql` in this directory is the canonical, hand-authored MySQL DDL
(with InnoDB engine, ENUM columns, composite indexes, and foreign keys)
meant to be run directly against a MySQL server by a DBA/migration tool.

This module provides the application-side read/write layer using
SQLAlchemy Core, targeting the SAME logical schema in a
dialect-portable way. In production you point it at MySQL
(`mysql+pymysql://user:pass@host/kpt2`); for local development or CI
without a MySQL server available, the identical code works against an
in-memory or file-based SQLite database. This mirrors a common
production pattern: a DBA-reviewed raw-SQL migration alongside an
ORM/Core layer that the application actually calls.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

merchants = Table(
    "merchants",
    metadata,
    Column("merchant_id", Integer, primary_key=True),
    Column("display_name", String(120), nullable=False, default=""),
    Column("mbi_offset_min", Float, nullable=False, default=0.0),
    Column("is_flagged", Boolean, nullable=False, default=False),
)

orders = Table(
    "orders",
    metadata,
    Column("order_id", Integer, primary_key=True, autoincrement=True),
    Column("merchant_id", Integer, ForeignKey("merchants.merchant_id"), nullable=False),
    Column("order_time", DateTime, nullable=False),
    Column("actual_ready_time", DateTime, nullable=False),
    Column("for_time", DateTime, nullable=False),
    Column("rider_arrival_time", DateTime, nullable=False),
    Column("actual_kpt_min", Float, nullable=False),
    Column("active_orders", Integer, nullable=False),
    Column("complexity", Float, nullable=False),
    Column("wait_cluster", Float, nullable=False),
    Column("kls", Float, nullable=False),
    Column("for_adj_kpt_min", Float, nullable=False),
    Column("historical_pattern", Float, nullable=False),
    Column("data_split", String(10), nullable=False, default="train"),
)

predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False),
    Column("model_name", String(20), nullable=False),
    Column("predicted_kpt", Float, nullable=False),
    Column("abs_error", Float, nullable=False),
)

drift_flags = Table(
    "drift_flags",
    metadata,
    Column("flag_id", Integer, primary_key=True, autoincrement=True),
    Column("merchant_id", Integer, ForeignKey("merchants.merchant_id"), nullable=False),
    Column("flag_type", String(30), nullable=False),
    Column("metric_value", Float, nullable=False),
)


def get_engine(connection_url: str) -> Engine:
    """Create a SQLAlchemy engine.

    Examples:
        MySQL (production):
            mysql+pymysql://kpt2_user:password@localhost:3306/kpt2
        SQLite (local/dev/CI, no server required):
            sqlite:///kpt2_local.db
            sqlite:///:memory:
    """
    return create_engine(connection_url)


def create_all_tables(engine: Engine) -> None:
    metadata.create_all(engine)


def store_pipeline_results(
    engine: Engine, df: pd.DataFrame, mbi: dict[int, float]
) -> None:
    """Bulk-insert merchants, orders, and per-model predictions from a
    completed pipeline run. Uses executemany-style bulk inserts (one
    round trip per table) rather than row-by-row inserts.
    """
    create_all_tables(engine)

    with engine.begin() as conn:
        merchant_rows = [
            {
                "merchant_id": int(m),
                "display_name": f"merchant_{m}",
                "mbi_offset_min": float(offset),
                "is_flagged": False,
            }
            for m, offset in mbi.items()
        ]
        conn.execute(merchants.delete())
        conn.execute(merchants.insert(), merchant_rows)

        order_rows = df.apply(
            lambda row: {
                "merchant_id": int(row["merchant_id"]),
                "order_time": row["order_time"],
                "actual_ready_time": row["actual_ready_time"],
                "for_time": row["for_time"],
                "rider_arrival_time": row["rider_arrival_time"],
                "actual_kpt_min": float(row["actual_kpt"]),
                "active_orders": int(row["active_orders"]),
                "complexity": float(row["complexity"]),
                "wait_cluster": float(row["wait_cluster"]),
                "kls": float(row["kls"]),
                "for_adj_kpt_min": float(row["for_adj_kpt"]),
                "historical_pattern": float(row["historical_pattern"]),
                "data_split": row["split"],
            },
            axis=1,
        ).tolist()

        conn.execute(orders.delete())
        result = conn.execute(orders.insert(), order_rows)

        # SQLite/MySQL both support inserted_primary_key per row for
        # single-row inserts; for bulk inserts we re-select ids by
        # matching (merchant_id, order_time) which is unique enough
        # for this simulated dataset.
        order_id_lookup = conn.execute(
            select(orders.c.order_id, orders.c.merchant_id, orders.c.order_time)
        ).fetchall()
        id_map = {(r.merchant_id, r.order_time): r.order_id for r in order_id_lookup}

        prediction_rows = []
        for i, row in df.reset_index(drop=True).iterrows():
            order_id = id_map[(int(row["merchant_id"]), row["order_time"])]
            for model_name, col in [
                ("baseline", "baseline_kpt"),
                ("crs", "crs_kpt"),
                ("shadow", "shadow_kpt"),
            ]:
                prediction_rows.append(
                    {
                        "order_id": order_id,
                        "model_name": model_name,
                        "predicted_kpt": float(row[col]),
                        "abs_error": float(abs(row["actual_kpt"] - row[col])),
                    }
                )

        conn.execute(predictions.delete())
        conn.execute(predictions.insert(), prediction_rows)


def query_mae_by_model(engine: Engine) -> pd.DataFrame:
    """The indexed rollup query referenced in schema.sql."""
    query = select(
        predictions.c.model_name,
        func.avg(predictions.c.abs_error).label("mae"),
        func.count().label("n"),
    ).group_by(predictions.c.model_name)

    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(query).fetchall(), columns=["model_name", "mae", "n"])


def query_flagged_merchants(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.DataFrame(
            conn.execute(select(merchants).where(merchants.c.is_flagged.is_(True))).fetchall(),
            columns=merchants.columns.keys(),
        )
