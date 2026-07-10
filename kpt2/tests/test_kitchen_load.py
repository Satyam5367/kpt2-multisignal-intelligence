import pandas as pd

from kpt2.kitchen_load import compute_kitchen_load


def test_kls_is_normalized_between_0_and_1():
    df = pd.DataFrame(
        {
            "active_orders": [0, 5, 10, 15],
            "complexity": [1.0, 2.0, 3.0, 5.0],
            "wait_cluster": [0.0, 0.2, 0.5, 1.0],
        }
    )
    df = compute_kitchen_load(df)

    assert df["kls"].min() == 0.0
    assert df["kls"].max() == 1.0
    assert (df["kls"] >= 0).all() and (df["kls"] <= 1).all()
    # Monotonic with increasing load inputs in this constructed example.
    assert df["kls"].is_monotonic_increasing


def test_kls_handles_constant_input_without_dividing_by_zero():
    df = pd.DataFrame(
        {
            "active_orders": [3, 3, 3],
            "complexity": [2.0, 2.0, 2.0],
            "wait_cluster": [0.1, 0.1, 0.1],
        }
    )
    df = compute_kitchen_load(df)
    assert (df["kls"] == 0.0).all()
