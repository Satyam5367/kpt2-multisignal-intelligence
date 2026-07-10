"""
End-to-end regression test.

This is deliberately opinionated: it asserts that CRS beats the raw
baseline and that Shadow beats the raw baseline too, on the HELD-OUT
test split. If someone reintroduces the original bug class (e.g. mixing
in uncorrelated random features with fixed weights), this test is what
catches it, the same way it should have caught the original project's
core flaw.
"""

from kpt2.config import SimulationConfig
from kpt2.evaluation import evaluate
from kpt2.pipeline import run_pipeline


def test_pipeline_runs_end_to_end_without_error():
    config = SimulationConfig(num_orders=400, num_merchants=15)
    result = run_pipeline(config)
    df = result["df"]

    required_columns = {
        "actual_kpt",
        "baseline_kpt",
        "crs_kpt",
        "shadow_kpt",
        "split",
    }
    assert required_columns.issubset(df.columns)
    assert len(df) == 400


def test_crs_beats_raw_baseline_on_held_out_split():
    config = SimulationConfig(num_orders=1200, num_merchants=30)
    result = run_pipeline(config)
    df = result["df"]

    metrics = evaluate(df, split="test").set_index("model")
    baseline_mae = metrics.loc["Baseline (raw FOR)", "mae"]
    crs_mae = metrics.loc["CRS", "mae"]
    shadow_mae = metrics.loc["Shadow", "mae"]

    assert crs_mae < baseline_mae, "CRS must outperform the raw merchant FOR signal"
    assert shadow_mae < baseline_mae, (
        "Shadow model (FOR-independent) must still beat the naive raw baseline"
    )


def test_evaluation_percentiles_are_ordered():
    config = SimulationConfig(num_orders=400, num_merchants=15)
    result = run_pipeline(config)
    metrics = evaluate(result["df"])
    for _, row in metrics.iterrows():
        assert row["p50_error"] <= row["p90_error"]
