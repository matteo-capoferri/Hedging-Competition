import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest.backtesting import price_and_deltas
from preprocessing.build_calendar import build_calendar
from pricing.make_collar_pricer_mm import make_collar_pricer_mm
from models.black76 import black_76


class MomentMatchingTests(unittest.TestCase):
    def test_single_remaining_fixing_matches_black(self):
        pricer = make_collar_pricer_mm(
            np.array([50.0]), np.array([False]), np.array([0.25]),
            (1.0, {"K": [1], "IV": [0]}),
            (60.0, {"K": [20, 100], "IV": [0.3, 0.3]}),
            dict(Kp=55, Kc=65, DF=0.97, correlation=0.8),
        )
        expected = 0.97 * 0.5 * (
            black_76(60, 60, 0.25, 0.3, "Put")
            - black_76(60, 80, 0.25, 0.3, "Call")
        )
        self.assertAlmostEqual(pricer(1, 60), expected)

    def test_fixings_and_deltas_without_simulation(self):
        contracts = ("COJ21", "COK21")
        for date in ["2021-01-15", "2021-02-01", "2021-02-24", "2021-02-25", "2021-02-26"]:
            today = pd.Timestamp(date)
            observed, future, _ = build_calendar(today, contracts)
            rows = {(row.date, row.future_ticker): 60.0 for row in observed.itertuples()}
            smiles = {}
            for contract in future["future_ticker"].unique():
                rows[(today, contract)] = 60.0
                smiles[contract] = {today: {"K": [20, 100], "IV": [0, 0]}}
            spot = pd.DataFrame({"settlement": list(rows.values())},
                                index=pd.MultiIndex.from_tuples(list(rows)))
            for put, call in [(100, 120), (0, 1)]:
                args = dict(Kp=put, Kc=call, DF=0.97, contracts=contracts, correlation=0.8)
                with patch("numpy.random.default_rng", side_effect=AssertionError("No simulation")):
                    result = price_and_deltas(today, "mm", args, spot, smiles)
                value = 0.97 * (max(put - 60, 0) - max(60 - call, 0))
                deltas = [-0.97 * future["future_ticker"].eq(c).sum() / 20 for c in contracts]
                np.testing.assert_allclose(result, [value, *deltas], atol=1e-9)
