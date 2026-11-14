from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models.extract_smile import extract_smile
from plot_smiles import plot_smiles
from preprocessing.build_calendar import build_calendar
from preprocessing.market_data import prepare_market_data
from pricing.calculate_collar_strike import calculate_collar_strike

DATA_DIR = Path(__file__).resolve().parent
EMISSION_DATE = pd.Timestamp("2020-01-06")
CONTRACTS = ("COJ21", "COK21")
CALL_STRIKE = 78


def main(data_dir=DATA_DIR):
    # 1. Prepare market data and calendar
    quotes, spot = prepare_market_data(data_dir, EMISSION_DATE, CONTRACTS)
    _, future, time_steps = build_calendar(EMISSION_DATE, CONTRACTS)

    # 2. Extract daily smiles
    smiles = extract_smile(quotes)

    # 3. Calculate the collar put strike
    collar_strike = calculate_collar_strike(
        spot, smiles, EMISSION_DATE, CONTRACTS, future, time_steps, CALL_STRIKE,
        n_paths=100_000, correlation=0.8, seed=42,
    )

    # 4. Display results
    print("Number of time steps:", len(time_steps))
    print("Collar put strike:", collar_strike)
    daily_smiles = {}
    for contract in CONTRACTS:
        daily_smiles[contract] = smiles[contract][EMISSION_DATE]
    plot_smiles(daily_smiles, EMISSION_DATE)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    main(parser.parse_args().data_dir)
