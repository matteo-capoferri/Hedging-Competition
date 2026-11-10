from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_smile import extract_smile
from get_initial_price import get_initial_price
from build_calendar import build_calendar
from simulate_paths import simulate_paths
from plot_smiles import plot_smiles
from find_K import find_K
from price_collar import price_collar 

DATA_DIR = Path(__file__).resolve().parent
EMISSION_DATE = pd.Timestamp("2020-01-06")


def main():
    quotes = pd.read_parquet(
        DATA_DIR / "unpivoted_bid_offer_train_clean.parquet",
        columns=["date", "actual_underlying_ticker", "strike", "option_type",
                 "settlement_price", "underlying_settlement", "time_to_expiry_days"],
    )
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes = quotes.loc[quotes["time_to_expiry_days"]>0]
    spot = pd.read_excel(DATA_DIR / "COJ21_COZ22_historical_series.xlsx")
    spot["date"] = pd.to_datetime(spot["date"])
    
    options_apr21 = quotes.loc[quotes["actual_underlying_ticker"] == "COJ21"]
    options_may21 = quotes.loc[quotes["actual_underlying_ticker"] == "COK21"]


    smiles = {}
    smiles["apr21"] = extract_smile(options_apr21)
    smiles["may21"] = extract_smile(options_may21)
    print(smiles["apr21"][EMISSION_DATE]["IV"])

    initial_price_apr21 = spot
    initial_price_may21 = get_initial_price(options_may21)
    volatility_apr21 = np.interp(initial_price_apr21, smiles["apr21"]["K"], smiles["apr21"]["IV"])
    volatility_may21 = np.interp(initial_price_may21, smiles["may21"]["K"], smiles["may21"]["IV"])
    fixing, time_steps = build_calendar(EMISSION_DATE)
    paths_apr21, paths_may21 = simulate_paths(
        initial_price_apr21, initial_price_may21,
        volatility_apr21, volatility_may21, time_steps,
        n_paths=100_000, correlation=0.8, seed=42,
    )
    print("Number of time steps:", len(fixing) - 1)
    print("Mean final April futures price:", paths_apr21[:, -3].mean())
    print("Mean final May futures price:", paths_may21[:, -1].mean())
    plot_smiles(smiles, EMISSION_DATE)
    plt.show()

    K_collar = find_K(paths_apr21, paths_may21, 78)
    P_C = price_collar(paths_apr21, paths_may21, K_collar, 78, 1)
    print(P_C)
if __name__ == "__main__":
    main()
