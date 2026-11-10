import numpy as np

from models.simulate_paths import simulate_paths
from pricing.find_K import find_K


def calculate_collar_strike(spot, smiles, date, contracts, time_steps, call_strike,
                            n_paths=100_000, correlation=0.8, seed=42):
    """
    INPUT:
        spot = daily futures prices, with one row per date and contract.
        smiles = dictionary organized as smiles[contract][date], with K and IV lists.
        date = market date used for initial futures prices and smiles.
        contracts = April and May futures tickers, in this order.
        time_steps = simulation intervals in years, starting from date.
                     The last two fixing dates use the May future.
        call_strike = fixed strike of the collar call.
        n_paths = number of simulated paths; default is 100000.
        correlation = correlation between futures shocks; default is 0.8.
        seed = random seed for repeatable results; default is 42.

    OUTPUT:
        collar_strike = put strike that matches the simulated call value.

    Select daily market inputs, simulate both futures, and calculate the put strike.
    """
    daily_spot = spot.loc[spot["date"] == date]

    price_april = daily_spot.loc[
        daily_spot["future_ticker"] == contracts[0], "settlement"
    ].item()
    price_may = daily_spot.loc[
        daily_spot["future_ticker"] == contracts[1], "settlement"
    ].item()

    smile_april = smiles[contracts[0]][date]
    smile_may = smiles[contracts[1]][date]

    paths_april, paths_may = simulate_paths(
        price_april,
        price_may,
        np.interp(price_april, smile_april["K"], smile_april["IV"]),
        np.interp(price_may, smile_may["K"], smile_may["IV"]),
        time_steps=time_steps,
        n_paths=n_paths,
        correlation=correlation,
        seed=seed,
    )
    collar_strike = find_K(paths_april, paths_may, call_strike)
    return collar_strike
