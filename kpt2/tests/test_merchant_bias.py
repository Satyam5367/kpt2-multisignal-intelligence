import pandas as pd

from kpt2.merchant_bias import apply_bias_correction, compute_merchant_bias_index


def _toy_df():
    base = pd.Timestamp("2026-01-01 12:00:00")
    return pd.DataFrame(
        {
            "merchant_id": [1, 1, 2, 2],
            "order_time": [base, base, base, base],
            "for_time": [
                base + pd.Timedelta(minutes=20),
                base + pd.Timedelta(minutes=22),
                base + pd.Timedelta(minutes=15),
                base + pd.Timedelta(minutes=17),
            ],
            "rider_arrival_time": [
                base + pd.Timedelta(minutes=25),
                base + pd.Timedelta(minutes=27),
                base + pd.Timedelta(minutes=14),
                base + pd.Timedelta(minutes=16),
            ],
        }
    )


def test_bias_index_is_mean_offset_per_merchant():
    df = _toy_df()
    mbi = compute_merchant_bias_index(df)

    assert mbi[1] == 5.0  # rider always arrives 5 min after FOR
    assert mbi[2] == -1.0  # rider arrives 1 min BEFORE FOR (early marking)


def test_bias_correction_shifts_for_time_by_merchant_offset():
    df = _toy_df()
    mbi = compute_merchant_bias_index(df)
    df = apply_bias_correction(df, mbi)

    # merchant 1: for_adj = for_time - 5min -> equals order_time + 15min
    assert df.loc[0, "for_adj_kpt"] == 15.0
    # merchant 2: for_adj = for_time - (-1min) = for_time + 1min
    assert df.loc[2, "for_adj_kpt"] == 16.0
