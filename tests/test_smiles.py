import unittest

import numpy as np
import pandas as pd

from models.black76 import black_76
from models.extract_smile import extract_smile


class SmileTests(unittest.TestCase):
    def test_contracts_dates_sorting_and_mean(self):
        rows = []
        for contract in ["COJ21", "COK21"]:
            for day, base_volatility in [("2020-01-07", 0.3), ("2020-01-06", 0.2)]:
                for strike, kind, volatility in [
                    (65, "Call", base_volatility),
                    (60, "Call", base_volatility),
                    (60, "Put", base_volatility + 0.1),
                ]:
                    rows.append({
                        "actual_underlying_ticker": contract,
                        "date": pd.Timestamp(day),
                        "strike": strike,
                        "option_type": kind,
                        "underlying_settlement": 60,
                        "time_to_expiry_days": 365,
                        "settlement_price": black_76(60, strike, 1, volatility, kind),
                    })
        smiles = extract_smile(pd.DataFrame(rows))
        self.assertEqual(set(smiles), {"COJ21", "COK21"})
        for contract in smiles:
            for day, expected in [("2020-01-06", [0.25, 0.2]), ("2020-01-07", [0.35, 0.3])]:
                smile = smiles[contract][pd.Timestamp(day)]
                self.assertEqual(smile["K"], [60, 65])
                np.testing.assert_allclose(smile["IV"], expected)


if __name__ == "__main__":
    unittest.main()
