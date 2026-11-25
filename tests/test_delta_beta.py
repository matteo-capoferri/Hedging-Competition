import unittest

import numpy as np
import pandas as pd

from models.estimate_delta_beta import estimate_delta_beta


class DeltaBetaTests(unittest.TestCase):
    def test_recovers_two_future_exposures_without_future_data(self):
        dates = pd.bdate_range("2020-01-01", periods=70)
        rng = np.random.default_rng(7)
        prices = pd.DataFrame(60 + rng.normal(size=(70, 2)).cumsum(axis=0),
                              index=dates, columns=["A", "B"])
        spot = prices.stack().rename("settlement").to_frame()
        quotes = pd.DataFrame({"date": dates, "actual_underlying_ticker": "C",
                               "underlying_settlement": 0.4 * prices.A.values + 0.7 * prices.B.values})
        quotes.loc[quotes.date > dates[60], "underlying_settlement"] = 10000
        weights = estimate_delta_beta(dates[60], "C", ("A", "B"), spot, quotes)
        np.testing.assert_allclose(weights, (0.4, 0.7), atol=1e-10)
        self.assertIsNone(estimate_delta_beta(dates[10], "C", ("A", "B"), spot, quotes))

    def test_native_future_needs_no_regression(self):
        self.assertEqual(estimate_delta_beta(None, "B", ("A", "B"), None, None), (0., 1.))
