import unittest

import pandas as pd

from preprocessing.options import filter_option_quotes, filter_smile_quotes


class QuoteFilterTests(unittest.TestCase):
    def test_liquidity_and_expiration(self):
        quotes = pd.DataFrame({
            "time_to_expiry_days": [1, 1, 1, 1, 1, 0, float("inf")],
            "option_bid": [0.75, float("nan"), 2, 0.02, 0.5, 1, 1],
            "option_ask": [1.25, 1, 1, 0.02, 1.5, 1, 1],
        })
        original = quotes.copy()

        result = filter_option_quotes(quotes)

        self.assertEqual(result.index.tolist(), [0])
        pd.testing.assert_frame_equal(quotes, original)

    def test_settlement_bounds_and_black_inputs(self):
        quotes = pd.DataFrame({
            "strike": [50, 70, 50, 50, 50, 50, 0, 50],
            "underlying_settlement": [60, 60, 60, 60, 60, 60, 60, float("inf")],
            "option_type": ["Call", "Put", "Call", "Call", "Call", "Other", "Call", "Call"],
            "settlement_price": [10, 11, 9, 60, float("nan"), 11, 11, 11],
        })

        self.assertEqual(filter_smile_quotes(quotes).index.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
