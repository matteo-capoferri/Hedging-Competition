import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest.backtesting import price_and_deltas
from models.simulate_paths import simulate_paths
from preprocessing.build_calendar import build_calendar
from pricing.select_vols import select_vols
from pricing.price_collar import price_collar


class SensTests(unittest.TestCase):
    def setUp(self):
        self.day = pd.Timestamp("2021-01-15")
        self.contracts = ("COJ21", "COK21")
        self.spot = pd.DataFrame(
            {"settlement": [60.0, 64.0]},
            index=pd.MultiIndex.from_tuples(
                [(self.day, contract) for contract in self.contracts]
            ),
        )
        self.smiles = {
            contract: {self.day: {"K": [40, 60, 80], "IV": vols}}
            for contract, vols in zip(self.contracts, [[0.4, 0.3, 0.25], [0.45, 0.35, 0.28]])
        }
        self.args = dict(Kp=58.0, Kc=65.0, DF=0.97, contracts=self.contracts,
                         n_paths=2000, seed=123, correlation=0.7, eps=1e-4)

    def test_sticky_moneyness_matches_full_resimulation(self):
        _, future, steps = build_calendar(self.day, self.contracts)
        mask = future["future_ticker"].eq(self.contracts[0]).to_numpy()
        forward = np.where(mask, 60.0, 64.0).mean()
        shocks = np.random.default_rng(123).standard_normal((2000, len(steps), 2))
        def price(april, may):
            values = []
            bumped_forward = np.where(mask, april, may).mean()
            # Independent interpolation on fixed moneyness nodes.
            vols = [tuple(np.interp(
                strike / bumped_forward,
                np.array(self.smiles[contract][self.day]["K"]) / reference,
                self.smiles[contract][self.day]["IV"],
            ) for contract, reference in zip(self.contracts, [60, 64]))
                for strike in [58, 65]]
            for strike, kind, pair in zip([58, 65], ["Put", "Call"], vols):
                a, m = simulate_paths(april, may, *pair, steps,
                                      n_paths=2000, correlation=0.7, shocks=shocks)
                average = np.where(mask, a, m).mean(axis=1)
                payoff = np.maximum(strike - average, 0) if kind == "Put" else np.maximum(average - strike, 0)
                values.append(0.97 * payoff.mean())
            return values[0] - values[1]

        expected = [(price(60.006, 64) - price(59.994, 64)) / 0.012,
                    (price(60, 64.0064) - price(60, 63.9936)) / 0.0128]
        with patch("pricing.simulate_leg.simulate_paths", wraps=simulate_paths) as simulator:
            value, april, may = price_and_deltas(self.day, "MonteCarlo", self.args, self.spot, self.smiles)
        self.assertEqual(simulator.call_count, 10)
        self.assertAlmostEqual(value, price(60.0, 64.0))
        np.testing.assert_allclose([april, may], expected, rtol=1e-9, atol=1e-10)
        self.assertTrue(april <= 0 and may <= 0)

    def test_unsupported_cases_are_explicit(self):
        with self.assertRaises(NotImplementedError):
            price_and_deltas(self.day, "MomentMatching", self.args, self.spot, self.smiles)

    def test_fixings_roll_and_fully_fixed(self):
        for day in ["2021-02-01", "2021-02-15", "2021-02-24", "2021-02-25", "2021-02-26"]:
            with self.subTest(day=day):
                today = pd.Timestamp(day)
                observed, future, _ = build_calendar(today, self.contracts)
                rows = {(row.date, row.future_ticker): 60.0
                        for row in observed.itertuples(index=False)}
                smiles = {}
                for contract in future["future_ticker"].unique():
                    rows[(today, contract)] = 60.0
                    smiles[contract] = {today: {"K": [20, 100], "IV": [0.0, 0.0]}}
                spot = pd.DataFrame({"settlement": list(rows.values())},
                                    index=pd.MultiIndex.from_tuples(list(rows)))
                args = dict(self.args, Kp=100.0, Kc=120.0)
                actual = price_and_deltas(today, "MonteCarlo", args, spot, smiles)
                expected = [-0.97 * future["future_ticker"].eq(c).sum() / 20
                            for c in self.contracts]
                np.testing.assert_allclose(actual, [0.97 * (100 - 60), *expected], atol=1e-10)
                # Non-positive effective strikes: zero put and linear call.
                actual = price_and_deltas(today, "MonteCarlo", dict(args, Kp=0, Kc=1), spot, smiles)
                np.testing.assert_allclose(actual, [-0.97 * (60 - 1), *expected], atol=1e-10)

    def test_price_collar_separate_legs_and_observed_fixings(self):
        value = price_collar([50], np.array([[40.0], [60.0]]), 55, 60, 0.9,
                             simulated_call_prices=np.array([[100.0], [80.0]]))
        self.assertAlmostEqual(value, -4.5)
        self.assertAlmostEqual(price_collar([50, 70], np.empty((1, 0)), 65, 80, 0.9), 4.5)


if __name__ == "__main__":
    unittest.main()
