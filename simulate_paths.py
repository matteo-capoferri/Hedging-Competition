import numpy as np


def simulate_paths(initial_price_apr21, initial_price_may21,
                   volatility_apr21, volatility_may21, time_steps,
                   n_paths=100_000, correlation=0.8, seed=42):
    """Simula due futures lognormali. Le matrici hanno righe per scenario e colonne per fixing."""
    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, len(time_steps), 2))
    shocks[:, :, 1] = correlation * shocks[:, :, 0] + np.sqrt(1 - correlation**2) * shocks[:, :, 1]

    prices = np.array([initial_price_apr21, initial_price_may21])
    vols = np.array([volatility_apr21, volatility_may21])
    steps = np.asarray(time_steps)[:, None]
    increments = -0.5 * vols**2 * steps + vols * np.sqrt(steps) * shocks
    paths = prices * np.exp(increments.cumsum(axis=1))
    return paths[:, :, 0], paths[:, :, 1]
