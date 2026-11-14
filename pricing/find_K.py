from scipy.optimize import brentq
import numpy as np
from pricing.price_collar import price_collar


def find_K(simulated_prices, Kc):
    return brentq(
        lambda Kp: price_collar([], simulated_prices, Kp, Kc, 1.0),
        1e-5,
        150,
    )
