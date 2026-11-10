import numpy as np
from scipy.optimize import brentq


def price_collar(paths_apr21, paths_may21, Kp, Kc, DF):
    return price_from_average(legacy_average(paths_apr21, paths_may21), Kp, Kc, DF)


def legacy_average(paths_apr21, paths_may21):
    """Compatibilità con le chiamate storiche che passano due matrici."""
    totals = paths_apr21[:, :-2].sum(axis=1) + paths_may21[:, -2:].sum(axis=1)
    return totals / paths_may21.shape[1]


def price_from_average(average, put_strike, call_strike, discount):
    """Value the desk's long put and short call on the same average."""
    payoff = np.maximum(put_strike - average, 0) - np.maximum(average - call_strike, 0)
    return discount * float(payoff.mean())


def zero_cost_put_strike(average, call_strike, *, lower=0.0, upper=None):
    def collar_value(put_strike):
        return price_from_average(average, put_strike, call_strike, discount=1.0)

    if upper is None:
        call_price = np.maximum(average - call_strike, 0).mean()
        upper = max(1.0, float(average.max()) + call_price + 1.0)
    return brentq(collar_value, lower, upper)
