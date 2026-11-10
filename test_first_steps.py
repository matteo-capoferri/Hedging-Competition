"""Controlli numerici e di integrita', senza dipendenze di test aggiuntive."""
import unittest

import numpy as np
import pandas as pd

from first_steps import Config, black76, futures_panel, implied_vol, prepare_quotes


class BlackTests(unittest.TestCase):
    def test_known_atm_value(self):
        self.assertAlmostEqual(black76(100, 100, 1, 0.2, "Call"), 7.9655674554, places=8)

    def test_put_call_parity_and_iv_recovery(self):
        for strike in (40, 60, 80):
            for maturity in (0.1, 1.5):
                for sigma in (0.2, 0.8):
                    call = black76(60, strike, maturity, sigma, "Call")
                    put = black76(60, strike, maturity, sigma, "Put")
                    self.assertAlmostEqual(call - put, 60 - strike, places=10)
                    kind = "Call" if strike >= 60 else "Put"
                    row = pd.Series(dict(underlying_settlement=60, strike=strike,
                                         ttm=maturity, option_type=kind,
                                         option_mid=call if kind == "Call" else put))
                    self.assertAlmostEqual(implied_vol(row), sigma, places=7)

    def test_intrinsic_and_invalid_inputs(self):
        self.assertEqual(black76(60, 50, 0, 0.3, "Call"), 10)
        self.assertEqual(black76(60, 70, 1, 0, "Put"), 10)
        for bad in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                black76(bad, 50, 1, 0.3, "Call")


class DataTests(unittest.TestCase):
    def test_filters_do_not_replace_missing_or_crossed_quotes(self):
        q = pd.DataFrame(dict(
            date=pd.to_datetime(["2020-01-06"] * 4),
            expiration=pd.to_datetime(["2021-01-06"] * 4),
            strike=[60.] * 4, underlying_settlement=[60.] * 4,
            option_type=["Call"] * 4, option_bid=[4., np.nan, 6., 0.],
            option_ask=[5., 5., 5., 0.],
        ))
        original = q.copy(deep=True)
        result = prepare_quotes(q, Config())
        self.assertEqual(result.eligible.tolist(), [True, False, False, False])
        self.assertTrue(np.isnan(result.option_mid.iloc[1]))
        pd.testing.assert_frame_equal(q, original)

    def test_history_extends_index_without_filling_gaps(self):
        q = pd.DataFrame(dict(
            date=pd.to_datetime(["2020-01-06", "2020-01-08"]),
            future_contract_code=["COJ21"] * 2,
            underlying_settlement=[60., 62.], underlying_bid=[59., 61.],
            underlying_ask=[61., 63.],
        ))
        history = {"COJ21": pd.DataFrame(dict(
            date=pd.to_datetime(["2020-01-03", "2020-01-06"]), settlement=[59., 60.],
        )), "COK21": pd.DataFrame(dict(
            date=pd.to_datetime(["2020-01-07"]), settlement=[61.],
        ))}
        panel = futures_panel(q, history)
        self.assertEqual(panel.loc["2020-01-03", "COJ21"], 59.)
        self.assertTrue(np.isnan(panel.loc["2020-01-07", "COJ21"]))
        history["COJ21"].loc[1, "settlement"] = 99.
        with self.assertRaises(ValueError):
            futures_panel(q, history)


if __name__ == "__main__":
    unittest.main()
