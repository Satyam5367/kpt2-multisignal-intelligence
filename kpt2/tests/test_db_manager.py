"""
The application-side db layer (db_manager.py) is dialect-portable
SQLAlchemy Core, so we validate it here against SQLite (no server
needed in CI). The exact same code targets MySQL in production by
changing only the connection URL -- see schema.sql and README for the
MySQL DDL and setup instructions.
"""

from kpt2.config import SimulationConfig
from kpt2.db.db_manager import (
    get_engine,
    query_mae_by_model,
    store_pipeline_results,
)
from kpt2.pipeline import run_pipeline


def test_store_and_query_round_trip():
    config = SimulationConfig(num_orders=200, num_merchants=8)
    result = run_pipeline(config)

    engine = get_engine("sqlite:///:memory:")
    store_pipeline_results(engine, result["df"], result["mbi"])

    mae_by_model = query_mae_by_model(engine).set_index("model_name")
    assert set(mae_by_model.index) == {"baseline", "crs", "shadow"}
    assert (mae_by_model["n"] == 200).all()
    assert mae_by_model.loc["crs", "mae"] < mae_by_model.loc["baseline", "mae"]
