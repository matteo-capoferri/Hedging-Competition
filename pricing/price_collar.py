import numpy as np

# def price_collar(paths_apr21, paths_may21, Kp, Kc, DF):
#     N = paths_may21.shape[1]
#     A = (paths_apr21[:,:-2].sum(axis=1) + paths_may21[:,-2:].sum(axis=1))/N    
#     mean_payoff = np.maximum(Kp-A, 0).mean() - np.maximum(A-Kc, 0).mean()
#     return mean_payoff * DF


def price_collar(observed_prices, simulated_prices, Kp, Kc, DF,
                 simulated_call_prices=None):
    """Price long put / short call including all observed fixings.

    simulated_prices contains put paths; optionally pass separate call paths
    when the legs use different smile volatilities. Both matrices contain
    only remaining fixings, in calendar order. An (n_paths, 0) matrix handles
    a fully fixed payoff. Existing callers can use one matrix for both legs.
    """
    if simulated_call_prices is None:
        simulated_call_prices = simulated_prices
    if simulated_prices.ndim != 2 or simulated_call_prices.shape != simulated_prices.shape:
        raise ValueError("Put and call paths must be matching two-dimensional matrices")
    total = len(observed_prices) + simulated_prices.shape[1]
    if total == 0 or simulated_prices.shape[0] == 0:
        raise ValueError("At least one fixing and one scenario are required")
    average = (np.sum(observed_prices) + simulated_prices.sum(axis=1)) / total
    call_average = (np.sum(observed_prices) + simulated_call_prices.sum(axis=1)) / total

    put = np.maximum(Kp - average, 0)
    call = np.maximum(call_average - Kc, 0)
    return DF * (put - call).mean()
