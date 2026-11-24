import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest.run_backtest import run_backtest
from backtest.hedge import rebalance_future
from backtest.check_accounting import check_accounting


class BacktestAccountingTests(unittest.TestCase):
    def run_scenario(self, valuations, prices, *, bids=None, asks=None, growth=None):
        dates = pd.date_range("2021-02-24", periods=len(prices))
        contracts = ("COJ21", "COK21")
        spot = pd.DataFrame({
            "settlement": prices,
            "bid": prices if bids is None else bids,
            "ask": prices if asks is None else asks,
        }, index=pd.MultiIndex.from_product([dates, [contracts[0]]]))
        sofr = None if growth is None else pd.DataFrame(
            {"known_rate": 0.0, "cash_growth": growth}, index=dates
        )
        # Diagnostiche solo per i test: osservano le vere esecuzioni senza
        # aggiungere colonne o logica di monitoraggio al ciclo del backtest.
        self.diagnostics = []

        def monitor_rebalance(position, target, bid, ask, settlement, lot_size):
            actual, pnl = rebalance_future(position, target, bid, ask, settlement, lot_size)
            self.diagnostics.append({
                "previous_position": position,
                "target": target,
                "trade": actual - position,
                "execution_pnl": pnl,
                "status": ("no_trade" if target == position else
                           "executed" if actual == target else "missing_quote"),
            })
            return actual, pnl

        with patch("backtest.run_backtest.price_and_deltas", side_effect=valuations), \
             patch("backtest.run_backtest.collar_vegas", return_value=(0.0, 0.0)), \
             patch("backtest.run_backtest.estimate_rho", return_value=0.8), \
             patch("backtest.run_backtest.rebalance_future", side_effect=monitor_rebalance):
            result = run_backtest(
                dates, dict(Kp=50, Kc=78, DF=1.0, contracts=contracts),
                spot, {}, barrels=1000, lot_size=1000,
                sofr=sofr, payment_date=dates[-1], method="mm",
            )
        # Ogni scenario controlla sia la riconciliazione giornaliera sia quella cumulata.
        np.testing.assert_allclose(result["accounting_error"], 0, atol=1e-8)
        self.assertAlmostEqual(result["delta_hedged_pnl"].sum(),
                               result["portfolio_value"].iloc[-1])
        return result

    def test_constant_market_has_zero_pnl(self):
        result = self.run_scenario([(0, -1, 0)] * 2, [60, 60])
        np.testing.assert_allclose(result["delta_hedged_pnl"], 0)
        self.assertEqual([d["status"] for d in self.diagnostics], ["executed", "no_trade"])

    def test_nonzero_initial_mark_is_included(self):
        result = self.run_scenario([(2, 0, 0)], [60])
        self.assertEqual(result["collar_pnl"].iloc[0], 2000)
        self.assertEqual(result["delta_hedged_pnl"].iloc[0], 2000)

    def test_previous_position_earns_move_before_closing(self):
        result = self.run_scenario([(0, -1, 0), (0, 0, 0)], [60, 61])
        self.assertEqual(result["futures_pnl"].iloc[1], 1000)
        self.assertEqual(result["position_april"].iloc[1], 0)

    def test_opening_and_closing_spreads(self):
        result = self.run_scenario(
            [(0, -1, 0), (0, 0, 0)], [60, 60], bids=[59.9, 59.9], asks=[60.1, 60.1]
        )
        np.testing.assert_allclose(result["execution_pnl"], [-100, -100])
        np.testing.assert_allclose([d["execution_pnl"] for d in self.diagnostics], [-100, -100])
        self.assertAlmostEqual(result["portfolio_value"].iloc[-1], -200)

    def test_negative_cash_pays_interest(self):
        result = self.run_scenario(
            [(0, -1, 0)] * 2, [60, 60], asks=[60.1, 60.1], growth=[1, 1.01]
        )
        self.assertAlmostEqual(result["interest_pnl"].iloc[1], -1)
        self.assertAlmostEqual(result["cash"].iloc[-1], -101)

    def test_missing_bid_preserves_position_and_marks_daily_move(self):
        result = self.run_scenario(
            [(0, -1, 0), (0, 0, 0), (0, 0, 0)],
            [60, 61, 62], bids=[60, np.nan, 62],
        )
        np.testing.assert_array_equal(result["position_april"], [1, 1, 0])
        # Il monitor dei test distingue un ordine saltato da una scelta di non negoziare.
        np.testing.assert_array_equal([d["trade"] for d in self.diagnostics], [1, 0, -1])
        np.testing.assert_allclose(result["futures_pnl"], [0, 1000, 1000])
        self.assertEqual(self.diagnostics[1]["status"], "missing_quote")
        self.assertEqual(self.diagnostics[1]["previous_position"], 1)
        self.assertEqual(self.diagnostics[1]["target"], 0)

    def test_terminal_payoff_is_counted_once_as_derivative_value(self):
        result = self.run_scenario([(0, 0, 0), (3, 0, 0)], [60, 60])
        self.assertEqual(result["cash"].iloc[-1], 0)
        self.assertEqual(result["portfolio_value"].iloc[-1], 3000)
        self.assertEqual(result["delta_hedged_pnl"].sum(), 3000)

    def test_nonfinite_valuation_fails_reconciliation(self):
        with self.assertRaisesRegex(ValueError, "riconcilia"):
            self.run_scenario([(np.nan, 0, 0)], [60])

    def test_accounting_monitor_rejects_inconsistent_pnl(self):
        # Verifica che il controllo attivo anche in produzione rilevi un errore reale.
        result = pd.DataFrame({"portfolio_value": [0.0, 1000.0],
                               "delta_hedged_pnl": [0.0, 900.0]})
        with self.assertRaisesRegex(ValueError, "riconcilia"):
            check_accounting(result)
