import unittest

import numpy as np
import pandas as pd

from build_calendar import build_fixing_schedule
from market_data import MarketData, build_future_quotes


class MarketDataTests(unittest.TestCase):
    def test_excel_resolves_only_confirmed_bid_ask(self):
        day = pd.Timestamp("2021-02-24")
        raw = pd.DataFrame({
            "date": [day] * 4,
            "future_contract_code": ["COJ21", "COJ21", "COK21", "COK21"],
            "underlying_settlement": [67., 67., 66., 66.],
            "underlying_bid": [66., 65., 65., 64.],
            "underlying_ask": [68., 69., 67., 68.],
        })
        history = {"COJ21": pd.DataFrame({
            "date": [day], "settlement": [67.], "bid": [65.], "ask": [69.],
        })}
        futures = build_future_quotes(raw, history)

        self.assertEqual(futures.loc[(day, "COJ21"), ["bid", "ask"]].tolist(), [65., 69.])
        self.assertEqual(futures.loc[(day, "COJ21"), "bid_ask_source"], "parquet+excel")
        self.assertEqual(futures.loc[(day, "COK21"), "settlement"], 66.)
        self.assertEqual(futures.loc[(day, "COK21"), "bid_ask_status"], "ambiguous")
        self.assertTrue(futures.loc[(day, "COK21"), ["bid", "ask"]].isna().all())

    def test_fixing_paths_follow_roll_and_ignore_future_observations(self):
        dates = pd.date_range("2021-02-22", periods=5)
        schedule = build_fixing_schedule(dates, "COJ21", "COK21", dates[-1])
        self.assertEqual(schedule.future_contract_code.tolist(), ["COJ21"] * 3 + ["COK21"] * 2)
        prices = schedule.assign(settlement=[10., 20., 30., 100., 100.])
        market = MarketData(prices.set_index(["date", "future_contract_code"]),
                            pd.DataFrame(), pd.Series(dtype=float))
        table = market.fixing_table(schedule, dates[1])
        self.assertEqual(table.status.tolist(), ["observed"] * 2 + ["pending"] * 3)
        self.assertTrue(table.loc[table.status.eq("pending"), "settlement"].isna().all())
        state = market.resolve_fixings(schedule, dates[1])
        np.testing.assert_allclose(
            state.average({"COJ21": np.array([[30., 999., 999.]]),
                           "COK21": np.array([[999., 40., 50.]])}),
            [30.],
        )
        np.testing.assert_allclose(market.resolve_fixings(schedule, dates[-1]).average({}), [52.])
