import unittest

import numpy as np
import pandas as pd

from backtest.backtesting import select_option
from models.black76 import black_76
from models.estimate_iv_beta import estimate_iv_beta


class IVBetaTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=65)
        changes = np.random.default_rng(42).normal(0, .001, 65)
        proxy_iv = .3 + changes.cumsum()
        self.quotes = pd.DataFrame([
            dict(date=day, actual_underlying_ticker=ticker, option_ticker=ticker,
                 strike=60., underlying_settlement=60., time_to_expiry_days=90,
                 expiration=day + pd.Timedelta(days=90), option_type="Call",
                 settlement_price=black_76(60, 60, 90/365, iv, "Call"),
                 option_bid=1., option_ask=1.1)
            for i, day in enumerate(self.dates)
            for ticker, iv in [("A", 2 * proxy_iv[i] - .1), ("B", proxy_iv[i])]
        ])

    def test_regression_direction_and_no_future_data(self):
        today = self.dates[-2]
        beta, r2 = estimate_iv_beta(today, "A", "B", self.quotes)
        self.assertAlmostEqual(beta, 2., places=6)
        self.assertAlmostEqual(r2, 1., places=6)
        changed = self.quotes.copy()
        changed.loc[changed.date > today, "settlement_price"] = 999.
        self.assertEqual(estimate_iv_beta(today, "A", "B", changed), (beta, r2))

    def test_insufficient_or_constant_history(self):
        self.assertEqual(estimate_iv_beta(self.dates[10], "A", "B", self.quotes), (None, None))
        self.quotes["settlement_price"] = black_76(60, 60, 90/365, .3, "Call")
        self.assertEqual(estimate_iv_beta(self.dates[-1], "A", "B", self.quotes), (None, None))

    def test_exclusions_do_not_mutate_history_or_keep_excluded_previous(self):
        quotes = self.quotes.copy()
        # Correlated future returns allow B to be ranked after excluding A.
        quotes["underlying_settlement"] += quotes.date.map(dict(zip(self.dates, np.sin(np.arange(65)))))
        original = quotes.copy()
        previous = quotes.loc[quotes.actual_underlying_ticker == "A"].iloc[-1]
        choice = select_option(self.dates[-1], "A", quotes, previous, excluded=["A"])
        self.assertEqual(choice.actual_underlying_ticker, "B")
        self.assertIsNone(select_option(self.dates[-1], "A", quotes, previous, excluded=["A", "B"]))
        pd.testing.assert_frame_equal(quotes, original)


if __name__ == "__main__":
    unittest.main()
