import numpy as np


def simulate_paths(initial_price_apr21, initial_price_may21,
                   volatility_apr21, volatility_may21, time_steps,
                   n_paths=100_000, correlation=0.8, seed=42):
    """
    INPUT:
        initial_price_apr21 = April futures price at the start of the simulation.
        initial_price_may21 = May futures price at the start of the simulation.
        volatility_apr21 = annualized April volatility, as a decimal.
        volatility_may21 = annualized May volatility, as a decimal.
        time_steps = array of time intervals between dates, in years.
        n_paths = number of simulated paths for each contract; default is 100000.
        correlation = correlation between the two random shocks; default is 0.8.
        seed = random generator seed for repeatable results; default is 42.

    OUTPUT:
        paths_apr21 = matrix of simulated April futures prices.
        paths_may21 = matrix of simulated May futures prices.
        Rows represent paths; columns represent future fixing dates.
        The initial prices are not included as a separate column.

    Simulate two futures price processes with constant volatilities and correlated shocks.
    """
    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, len(time_steps), 2))
    noise_apr21 = shocks[:, :, 0]
    noise_may21 = correlation * noise_apr21 + np.sqrt(1 - correlation**2) * shocks[:, :, 1]
    increments_apr21 = (
        -0.5 * volatility_apr21**2 * time_steps
        + volatility_apr21 * np.sqrt(time_steps) * noise_apr21
    )
    increments_may21 = (
        -0.5 * volatility_may21**2 * time_steps
        + volatility_may21 * np.sqrt(time_steps) * noise_may21
    )
    log_returns_apr21 = np.cumsum(increments_apr21, axis=1)
    log_returns_may21 = np.cumsum(increments_may21, axis=1)
    paths_apr21 = initial_price_apr21 * np.exp(log_returns_apr21)
    paths_may21 = initial_price_may21 * np.exp(log_returns_may21)
    return paths_apr21, paths_may21


def mean_fixing_prices(paths_april, paths_may):
    """Restituisce la media dei percorsi all'ultimo fixing di ogni contratto."""
    return paths_april[:, -3].mean(), paths_may[:, -1].mean()
