from pathlib import Path
import unittest

import pandas as pd

from preprocessing.sofr import discount_factor, prepare_sofr


SOFR_FILE = Path(__file__).resolve().parents[1] / "data" / "SOFR.csv"


class SofrTest(unittest.TestCase):
    def test_march_rate_cut_is_used_only_after_publication(self):
        dates = pd.to_datetime(["2020-03-13", "2020-03-16", "2020-03-17"])
        sofr = prepare_sofr(SOFR_FILE, dates)
        friday, monday, tuesday = dates
        self.assertEqual(sofr.at[friday, "cash_growth"], 1)
        self.assertAlmostEqual(sofr.at[monday, "known_rate"], 0.011)
        self.assertAlmostEqual(sofr.at[tuesday, "known_rate"], 0.0026)
        self.assertAlmostEqual(sofr.at[monday, "cash_growth"], 1 + 0.011 * 3 / 360)
        self.assertAlmostEqual(sofr.at[tuesday, "cash_growth"], 1 + 0.0026 / 360)
        self.assertEqual(discount_factor(tuesday, tuesday, 0.0026), 1)

    def test_brent_session_during_us_holiday(self):
        dates = pd.to_datetime(["2020-01-17", "2020-01-20", "2020-01-21"])
        sofr = prepare_sofr(SOFR_FILE, dates)
        friday, holiday, tuesday = dates
        self.assertEqual(sofr.at[holiday, "known_rate"], sofr.at[friday, "known_rate"])
        self.assertAlmostEqual(sofr.at[tuesday, "known_rate"], 0.0154)
        total_growth = sofr.at[holiday, "cash_growth"] * sofr.at[tuesday, "cash_growth"]
        self.assertAlmostEqual(total_growth, 1 + 0.0154 * 4 / 360)
