import numpy as np

from models.simulate_paths import simulate_paths
from pricing.select_vols import select_vols


def simulate_leg(effective_strike, prices, april_market, may_market,
                  uses_april, time_steps, correlation, shocks):
    """Simula i fixing residui della put o della call al suo strike equivalente."""
    april, may = prices
    forward_prices = np.where(uses_april, april, may)
    if effective_strike <= 0:
        # Put nulla o call lineare: bastano i prezzi attesi.
        return np.broadcast_to(forward_prices, shocks.shape[:2])

    ref_april, smile_april = april_market
    ref_may, smile_may = may_market
    vol_april, vol_may = select_vols(
        effective_strike, ref_april, ref_may, forward_prices.mean(),
        smile_april, smile_may,
    )
    paths_april, paths_may = simulate_paths(
        april, may, vol_april, vol_may, time_steps,
        n_paths=len(shocks), correlation=correlation, shocks=shocks,
    )
    return np.where(uses_april, paths_april, paths_may)
