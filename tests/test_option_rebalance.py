import unittest
from unittest.mock import patch

import pandas as pd

from backtest.run_backtest import run_backtest


class OptionRebalanceTests(unittest.TestCase):
    def run_case(self, missing_close=False):
        dates = pd.date_range("2021-01-01", periods=3)
        quotes = pd.DataFrame([
            dict(date=day, option_ticker=ticker, actual_underlying_ticker="A",
                 expiration=pd.Timestamp("2021-04-01"), strike=60 if ticker == "old" else 65, option_type="Call",
                 time_to_expiry_days=90, underlying_settlement=60,
                 settlement_price=4, option_bid=3.9, option_ask=4.1)
            for day in dates for ticker in ("old", "new")
            if not (missing_close and day == dates[2] and ticker == "old")
        ])
        selections = []
        for day, ticker in zip(dates, ("old", "old", "new")):
            selections.extend([quotes.loc[(quotes.date == day) & (quotes.option_ticker == ticker)].iloc[0], None])
        spot = pd.DataFrame(dict(settlement=60., bid=60., ask=60.),
                            index=pd.MultiIndex.from_product([dates, ["A", "B"]]))
        with patch("backtest.run_backtest.select_option", side_effect=selections), \
             patch("backtest.run_backtest.price_and_deltas", return_value=(0., 0., 0.)), \
             patch("backtest.run_backtest.collar_vegas", side_effect=[(-2., 0.), (-3., 0.), (-3., 0.)]), \
             patch("backtest.run_backtest.implied_volatility", return_value=.3), \
             patch("backtest.run_backtest.black_greeks",
                   side_effect=lambda future, strike, time, vol, kind: (.5 if strike == 60 else .8, 10.)):
            return run_backtest(dates, dict(contracts=("A", "B"), DF=1.), spot, {}, quotes=quotes)

    def test_same_ticker_adjusts_and_new_ticker_closes_then_opens(self):
        result = self.run_case()
        self.assertEqual(result.held_april_quantity.tolist(), [200, 300, 300])
        self.assertEqual(result.held_april_ticker.tolist(), ["old", "old", "new"])
        for actual, expected in zip(result.option_april_execution_pnl, [-20000, -10000, -60000]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue((result.cash == 0).all())  # Step 6 is deliberately unchanged.
        self.assertEqual(result.position_april.tolist(), [-100, -150, -240])
        self.assertTrue((result.position_may == 0).all())

    def test_failed_close_does_not_open_replacement(self):
        result = self.run_case(missing_close=True)
        self.assertEqual(result.iloc[-1].held_april_ticker, "old")
        self.assertEqual(result.iloc[-1].held_april_quantity, 300)
        self.assertEqual(result.iloc[-1].option_april_execution_pnl, 0)
        self.assertEqual(result.iloc[-1].position_april, -150)  # Old delta, not replacement delta.


if __name__ == "__main__":
    unittest.main()
