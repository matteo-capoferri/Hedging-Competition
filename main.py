from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest.run_backtest import run_backtest
from models.estimate_rho import estimate_rho
from models.extract_smile import extract_smile
from plot_smiles import plot_smiles
from preprocessing.build_calendar import build_calendar
from preprocessing.market_data import prepare_market_data
from preprocessing.sofr import prepare_sofr
from pricing.calculate_collar_strike import calculate_collar_strike
from pricing.calculate_collar_strike_mm import calculate_collar_strike_mm

DATA_DIR = Path(__file__).resolve().parent
SOFR_FILE = DATA_DIR / "data" / "SOFR.csv"
if not (DATA_DIR / "unpivoted_bid_offer_train_clean.parquet").is_file():
    if (DATA_DIR.parent / "unpivoted_bid_offer_train_clean.parquet").is_file():
        DATA_DIR = DATA_DIR.parent
EMISSION_DATE = pd.Timestamp("2020-01-06")
CONTRACTS = ("COJ21", "COK21")
CALL_STRIKE = 78


def main(data_dir=DATA_DIR):
    # 1. Prepare market data and calendar
    quotes, spot = prepare_market_data(data_dir, EMISSION_DATE, CONTRACTS)
    rho = estimate_rho(EMISSION_DATE, spot, CONTRACTS)
    _, future, time_steps = build_calendar(EMISSION_DATE, CONTRACTS)

    # 2. Extract daily smiles
    smiles = extract_smile(quotes)

    # 3. Calculate the collar put strike
    collar_strike = calculate_collar_strike(
        spot.at[(EMISSION_DATE, CONTRACTS[0]), "settlement"],
        spot.at[(EMISSION_DATE, CONTRACTS[1]), "settlement"],
        smiles, EMISSION_DATE, CONTRACTS, future, time_steps, CALL_STRIKE,
        n_paths=100_000, correlation=rho, seed=42,
    )
    collar_strike_mm = calculate_collar_strike_mm(
        spot.at[(EMISSION_DATE, CONTRACTS[0]), "settlement"],
        spot.at[(EMISSION_DATE, CONTRACTS[1]), "settlement"],
        smiles, EMISSION_DATE, CONTRACTS, future, time_steps, CALL_STRIKE,
        correlation=rho,
    )

    # 4. Display results
    print("Number of time steps:", len(time_steps))
    print("Futures correlation at issue:", rho)
    print("Collar put strike (Monte Carlo):", collar_strike)
    print("Collar put strike (moment matching):", collar_strike_mm)
    daily_smiles = {}
    for contract in CONTRACTS:
        daily_smiles[contract] = smiles[contract][EMISSION_DATE]
    # 5. Tutte le sedute dall'emissione all'ultimo fixing, inclusi.
    last_fixing = future["date"].max()
    market_dates = spot.index.get_level_values("date").unique().sort_values()
    dates = market_dates[(market_dates >= EMISSION_DATE) & (market_dates <= last_fixing)]
    sofr = prepare_sofr(SOFR_FILE, dates)
    backtest = run_backtest(
        dates,
        {"Kp": collar_strike, "Kc": CALL_STRIKE, "contracts": CONTRACTS},
        spot, smiles, sofr=sofr, payment_date=last_fixing,
    )
    # 6. Media dei P&L giornalieri al quadrato, inclusi i costi di apertura.
    hedged_score = backtest["daily_pnl"].pow(2).mean(skipna=False)
    unhedged_score = backtest["collar_pnl"].pow(2).mean(skipna=False)
    print(f"Sedute: {len(backtest)} (dal {dates[0]:%d/%m/%Y} al {dates[-1]:%d/%m/%Y})")
    print(f"Punteggio con copertura e interessi SOFR: {hedged_score:,.2f} USD²")
    print(f"Punteggio senza copertura: {unhedged_score:,.2f} USD²")

    plot_smiles(daily_smiles, EMISSION_DATE)
    plt.show()
    return backtest


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    main(parser.parse_args().data_dir)
