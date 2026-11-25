import unittest
from unittest.mock import patch

import pandas as pd

from backtest.run_backtest import run_backtest
from models.estimate_delta_beta import estimate_delta_beta


class OptionRebalanceTests(unittest.TestCase):
    def run_case(self, missing_close=False, price_step=0, cash_growth=None, proxy=False, iv_beta=1.):
        dates = pd.date_range("2021-01-01", periods=3)
        quotes = pd.DataFrame([
            dict(date=day, option_ticker=ticker, actual_underlying_ticker="A",
                 expiration=pd.Timestamp("2021-04-01"), strike=60 if ticker == "old" else 65, option_type="Call",
                 time_to_expiry_days=90, underlying_settlement=60,
                 settlement_price=4 + i * price_step,
                 option_bid=3.9 + i * price_step, option_ask=4.1 + i * price_step)
            for i, day in enumerate(dates) for ticker in ("old", "new")
            if not (missing_close and day == dates[2] and ticker == "old")
        ])
        if proxy:
            quotes.loc[quotes.option_ticker == "new", "actual_underlying_ticker"] = "B" if proxy is True else proxy
        selections = []
        for day, ticker in zip(dates, ("old", "old", "new")):
            selections.extend([quotes.loc[(quotes.date == day) & (quotes.option_ticker == ticker)].iloc[0], None])
        spot = pd.DataFrame(dict(settlement=60., bid=60., ask=60.),
                            index=pd.MultiIndex.from_product([dates, ["A", "B"]]))
        with patch("backtest.hedge.estimate_delta_beta", side_effect=lambda day, underlying, contracts, spot, quotes:
                   (.4, .7) if underlying == "C" else estimate_delta_beta(day, underlying, contracts, spot, quotes)), \
             patch("backtest.backtesting.select_option", side_effect=selections), \
             patch("backtest.backtesting.estimate_delta_beta", side_effect=lambda day, underlying, contracts, spot, quotes:
                   (.4, .7) if underlying == "C" else estimate_delta_beta(day, underlying, contracts, spot, quotes)), \
             patch("backtest.backtesting.estimate_iv_beta", side_effect=lambda day, target, underlying, quotes:
                   (iv_beta if target != underlying else 1., 1.)), \
             patch("backtest.backtesting.price_and_deltas", return_value=(0., 0., 0.)), \
             patch("backtest.backtesting.collar_vegas", side_effect=[(-2., 0.), (-3., 0.), (-3., 0.)]), \
             patch("backtest.backtesting.implied_volatility", return_value=.3), \
             patch("backtest.backtesting.black_greeks",
                   side_effect=lambda future, strike, time, vol, kind: (.5 if strike == 60 else .8, 10.)):
            sofr = None if cash_growth is None else pd.DataFrame(
                dict(known_rate=0., cash_growth=cash_growth), index=dates)
            return run_backtest(dates, dict(contracts=("A", "B"), DF=1.), spot, {}, quotes=quotes,
                                sofr=sofr, payment_date=dates[-1])

    def test_same_ticker_adjusts_and_new_ticker_closes_then_opens(self):
        result = self.run_case()
        self.assertEqual(result.held_april_quantity.tolist(), [200, 300, 300])
        self.assertEqual(result.held_april_ticker.tolist(), ["old", "old", "new"])
        for actual, expected in zip(result.option_april_execution_pnl, [-20000, -10000, -60000]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result.cash, [-20000, -30000, -90000]):
            self.assertAlmostEqual(actual, expected)
        self.assertLess(result.accounting_error.abs().max(), 1e-6)
        self.assertEqual(result.position_april.tolist(), [-100, -150, -240])
        self.assertTrue((result.position_may == 0).all())

    def test_failed_close_does_not_open_replacement(self):
        result = self.run_case(missing_close=True)
        self.assertEqual(result.iloc[-1].held_april_ticker, "old")
        self.assertEqual(result.iloc[-1].held_april_quantity, 300)
        self.assertEqual(result.iloc[-1].option_april_execution_pnl, 0)
        self.assertEqual(result.iloc[-1].position_april, -150)  # Old delta, not replacement delta.

    def test_mtm_uses_previous_quantities_even_on_replacement_day(self):
        result = self.run_case(price_step=1)
        self.assertEqual(result.options_pnl.tolist(), [0, 200000, 300000])
        for actual, expected in zip(result.cash, [-20000, 170000, 410000]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.hedged_pnl.sum(), result.portfolio_value.iloc[-1])

    def test_interest_applies_to_previous_cash_not_todays_flows(self):
        result = self.run_case(price_step=1, cash_growth=[1, 1.01, 1.01])
        self.assertAlmostEqual(result.interest_pnl.iloc[1], -200)
        self.assertAlmostEqual(result.interest_pnl.iloc[2], 1698)

    def test_stale_mark_does_not_repeat_previous_gain(self):
        result = self.run_case(missing_close=True, price_step=1)
        self.assertEqual(result.options_pnl.tolist(), [0, 200000, 0])

    def test_proxy_opens_and_its_delta_is_hedged_on_its_own_future(self):
        result = self.run_case(proxy=True, iv_beta=2.)
        self.assertEqual(result.iloc[-1].held_april_ticker, "new")
        self.assertEqual(result.iloc[-1].held_april_quantity, 600)
        self.assertEqual(result.iloc[-1].position_april, 0)
        self.assertEqual(result.iloc[-1].position_may, -480)
        self.assertAlmostEqual(result.iloc[-1].option_april_execution_pnl, -90000)
        self.assertLess(result.accounting_error.abs().max(), 1e-6)

    def test_third_underlying_delta_is_split_across_both_futures(self):
        result = self.run_case(proxy="C", iv_beta=2.)
        self.assertEqual(result.iloc[-1].held_april_quantity, 600)
        self.assertEqual(result.iloc[-1].position_april, -192)
        self.assertEqual(result.iloc[-1].position_may, -336)
        self.assertAlmostEqual(result.iloc[-1].option_execution_pnl, -90000)
        self.assertLess(result.accounting_error.abs().max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
