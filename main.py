from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_smile import extract_smile
from find_K import find_K
from plot_smiles import plot_smiles
from price_collar import price_collar
from preprocessing.build_calendar import build_calendar
from preprocessing.market_data import prepare_market_data
from simulate_paths import mean_fixing_prices, simulate_paths

DATA_DIR = Path(__file__).resolve().parent
EMISSION_DATE = pd.Timestamp("2020-01-06")
CONTRACTS = ("COJ21", "COK21")
CALL_STRIKE = 78


def main(data_dir=DATA_DIR):
    # 1. Preprocessing di dati e calendario
    quotes, spot = prepare_market_data(data_dir, EMISSION_DATE, CONTRACTS)
    _, time_steps = build_calendar(EMISSION_DATE)

    # 2. Smile e condizioni iniziali
    smiles = extract_smile(quotes)
    daily_smiles = {}
    prices = []
    volatilities = []
    for contract in CONTRACTS:
        daily_smiles[contract] = smiles[contract][EMISSION_DATE]
        daily_spot = spot.loc[
            (spot["date"] == EMISSION_DATE) & (spot["future_ticker"] == contract)
        ]
        price = daily_spot["settlement"].drop_duplicates().item()
        volatility = np.interp(price, daily_smiles[contract]["K"], daily_smiles[contract]["IV"])
        prices.append(price)
        volatilities.append(volatility)

    # 3. Simulazione e collar
    paths_april, paths_may = simulate_paths(
        *prices, *volatilities, time_steps=time_steps,
        n_paths=100_000, correlation=0.8, seed=42,
    )
    collar_strike = find_K(paths_april, paths_may, CALL_STRIKE)
    collar_price = price_collar(paths_april, paths_may, collar_strike, CALL_STRIKE, 1)

    # 4. Risultati
    mean_april, mean_may = mean_fixing_prices(paths_april, paths_may)
    print("Number of time steps:", len(time_steps))
    print("Mean final April futures price:", mean_april)
    print("Mean final May futures price:", mean_may)
    print(collar_price)
    plot_smiles(daily_smiles, EMISSION_DATE)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    main(parser.parse_args().data_dir)
