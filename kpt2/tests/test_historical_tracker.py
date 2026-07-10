from kpt2.historical_tracker import HistoricalPatternTracker


def test_first_observation_returns_itself_as_prediction():
    tracker = HistoricalPatternTracker(window_size=3)
    prediction = tracker.predict_and_update(merchant_id=1, observed_value=20.0)
    assert prediction == 20.0


def test_prediction_never_uses_current_observation():
    tracker = HistoricalPatternTracker(window_size=10)
    tracker.predict_and_update(1, 10.0)
    tracker.predict_and_update(1, 20.0)
    # Rolling average of [10, 20] = 15, BEFORE incorporating 1000.0
    prediction = tracker.predict_and_update(1, 1000.0)
    assert prediction == 15.0


def test_window_evicts_oldest_entry():
    tracker = HistoricalPatternTracker(window_size=2)
    tracker.predict_and_update(1, 10.0)
    tracker.predict_and_update(1, 20.0)
    # window is now [10, 20]; next call should average only these two
    prediction = tracker.predict_and_update(1, 999.0)
    assert prediction == 15.0
    # window is now [20, 999] (10 evicted); next call averages those two
    prediction2 = tracker.predict_and_update(1, 5.0)
    assert prediction2 == (20.0 + 999.0) / 2


def test_merchants_are_tracked_independently():
    tracker = HistoricalPatternTracker(window_size=5)
    tracker.predict_and_update(1, 100.0)
    prediction_for_merchant_2 = tracker.predict_and_update(2, 5.0)
    assert prediction_for_merchant_2 == 5.0  # unaffected by merchant 1's history
