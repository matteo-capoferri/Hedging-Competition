"""Price the training collar using a market snapshot and explicit fixing dates."""

import argparse
from pathlib import Path

import pandas as pd

from build_calendar import training_fixing_schedule
from market_data import MarketData, default_data_dir
from option_analytics import Config, atm_smile, daily_iv_panel, prepare_quotes
from price_collar import price_from_average, zero_cost_put_strike
from simulate_paths import simulate_paths


EMISSION_DATE = pd.Timestamp("2020-01-06")
CALL_STRIKE = 78.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--correlation", type=float, default=0.8,
                        help="Scenario rho at inception (default: 0.8)")
    parser.add_argument("--plot", action="store_true", help="Plot the input IVs (requires matplotlib)")
    args = parser.parse_args()

    market = MarketData.load(args.data_dir)
    schedule = training_fixing_schedule()
    fixings = market.resolve_fixings(schedule, EMISSION_DATE)
    legs = schedule.future_contract_code.unique()
    futures_today = market.future_quotes.xs(EMISSION_DATE, level="date")
    options_today = market.option_quotes.xs(
        EMISSION_DATE, level="date", drop_level=False,
    ).reset_index()
    options_today = options_today.loc[options_today.future_contract_code.isin(legs)]
    ivs = daily_iv_panel(prepare_quotes(options_today, Config()))

    initial_prices = futures_today.loc[legs, "settlement"].to_numpy()
    volatilities = []
    smiles = {}
    for code, forward in zip(legs, initial_prices, strict=True):
        points, vol = atm_smile(ivs, code, forward)
        volatilities.append(vol)
        smiles[code] = {"K": points.index.tolist(), "IV": points.tolist()}
        print(f"{code}: settlement {forward:.2f}, {len(points)} punti IV, vol ATM {vol:.2%}")

    dates = pd.DatetimeIndex(fixings.remaining.date).insert(0, EMISSION_DATE)
    time_steps = dates.to_series().diff().dropna().dt.days.to_numpy() / 365.0
    def simulate_average(seed):
        paths = simulate_paths(
            *initial_prices, *volatilities, time_steps, correlation=args.correlation, seed=seed,
        )
        return fixings.average(dict(zip(legs, paths, strict=True)))

    average = simulate_average(seed=42)
    put_strike = zero_cost_put_strike(average, CALL_STRIKE)
    price = price_from_average(average, put_strike, CALL_STRIKE, discount=1.0)

    validation_average = simulate_average(seed=43)
    validation = price_from_average(validation_average, put_strike, CALL_STRIKE, discount=1.0)
    print(f"Fixing (calendario feriale provvisorio): {schedule.future_contract_code.value_counts().to_dict()}")
    print(f"Correlazione di scenario: {args.correlation:.3f}")
    print(f"K_put: {put_strike:.4f} USD/bbl | K_call: {CALL_STRIKE:.2f} USD/bbl")
    print(f"Valore iniziale non scontato: {price:.6f} USD/bbl")
    print(f"Valore su simulazione indipendente: {validation:.6f} USD/bbl")

    if args.plot:
        import matplotlib.pyplot as plt
        from plot_smiles import plot_smiles

        plot_smiles(smiles, EMISSION_DATE)
        plt.show()


if __name__ == "__main__":
    main()
