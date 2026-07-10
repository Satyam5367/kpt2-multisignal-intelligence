"""
Historical pattern tracking via O(1)-amortized rolling windows.

For every merchant we maintain a bounded `collections.deque` of their
most recent bias-corrected KPT values. A deque with `maxlen` gives O(1)
append and automatic eviction of the oldest entry, which is the
standard efficient data structure for a fixed-size rolling window (a
naive list-based approach would be O(n) per update to trim old entries).

Critically, each order's `historical_pattern` value is computed from
values that occurred STRICTLY BEFORE it in time (a running/expanding
lookback), so this signal never leaks the current order's own outcome
-- it approximates what a real-time production system would have known
at prediction time.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from .config import SimulationConfig


class HistoricalPatternTracker:
    """Per-merchant rolling average of past bias-corrected KPT values."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._history: dict[int, deque] = {}

    def predict_and_update(self, merchant_id: int, observed_value: float) -> float:
        """Return the rolling average BEFORE incorporating `observed_value`,
        then push `observed_value` into the window for future calls.
        """
        window = self._history.setdefault(merchant_id, deque(maxlen=self.window_size))
        prediction = sum(window) / len(window) if window else observed_value
        window.append(observed_value)
        return prediction


def compute_historical_pattern(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Walk orders in chronological order (globally) and, for each
    merchant, emit a leakage-free rolling-average signal based on that
    merchant's own past bias-corrected KPT values.
    """
    tracker = HistoricalPatternTracker(config.history_window_size)
    ordered_idx = df.sort_values("order_time").index

    historical_pattern = pd.Series(index=df.index, dtype=float)
    for idx in ordered_idx:
        merchant_id = int(df.at[idx, "merchant_id"])
        observed = float(df.at[idx, "for_adj_kpt"])
        historical_pattern.at[idx] = tracker.predict_and_update(merchant_id, observed)

    df["historical_pattern"] = historical_pattern
    return df
