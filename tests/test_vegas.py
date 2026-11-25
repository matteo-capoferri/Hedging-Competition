import copy
import unittest

import numpy as np
import pandas as pd

from sensitivities.price_and_deltas import price_and_deltas
from preprocessing.build_calendar import build_calendar
from sensitivities.collar_vegas import collar_vegas
from models.black76 import black_76


class VegaTests(unittest.TestCase):
    def test_parallel_shift_matches_repricing_without_mutating_smiles(self):
        day = pd.Timestamp("2021-01-15")
        contracts = ("COJ21", "COK21")
        spot = pd.DataFrame({"settlement": [60.0, 64.0]},
                            index=pd.MultiIndex.from_tuples([(day, c) for c in contracts]))
        smiles = {c: {day: {"K": [40, 60, 80], "IV": [0.4, 0.3, 0.25]}} for c in contracts}
        original = copy.deepcopy(smiles)
        args = dict(Kp=58, Kc=65, DF=0.97, contracts=contracts, correlation=0.7,
                    n_paths=1000, seed=123)
        for method in ("mm", "MonteCarlo"):
            vegas = collar_vegas(day, method, args, spot, smiles)
            base = price_and_deltas(day, method, args, spot, smiles)[0]
            for index, contract in enumerate(contracts):
                shifted = copy.deepcopy(smiles)
                shifted[contract][day]["IV"] = [vol + 0.0001 for vol in shifted[contract][day]["IV"]]
                bumped = price_and_deltas(day, method, args, spot, shifted)[0]
                self.assertAlmostEqual(vegas[index], (bumped - base) / 0.0001)
        self.assertEqual(smiles, original)

    def test_last_fixing_matches_black_and_expired_vegas_are_zero(self):
        contracts = ("COJ21", "COK21")
        day = pd.Timestamp("2021-02-25")
        observed, _, _ = build_calendar(day, contracts)
        rows = {(row.date, row.future_ticker): 60.0 for row in observed.itertuples()}
        spot = pd.DataFrame({"settlement": list(rows.values())},
                            index=pd.MultiIndex.from_tuples(list(rows)))
        smiles = {"COK21": {day: {"K": [20, 100], "IV": [0.3, 0.3]}}}
        args = dict(Kp=60, Kc=61, DF=0.97, contracts=contracts, correlation=0.8)
        vegas = collar_vegas(day, "mm", args, spot, smiles)
        values = []
        for vol in (0.3, 0.3001):
            values.append(0.97 / 20 * (
                black_76(60, 60, 1 / 365, vol, "Put")
                - black_76(60, 80, 1 / 365, vol, "Call")
            ))
        self.assertEqual(vegas[0], 0)
        self.assertAlmostEqual(vegas[1], (values[1] - values[0]) / 0.0001)
        for method in ("mm", "MonteCarlo"):
            self.assertEqual(collar_vegas(pd.Timestamp("2021-02-26"), method, args, spot, {}),
                             (0.0, 0.0))
