"""Interfaccia storica per la ricerca dello strike put."""

from price_collar import legacy_average, zero_cost_put_strike


def find_K(paths_apr21, paths_may21, Kc):
    average = legacy_average(paths_apr21, paths_may21)
    return zero_cost_put_strike(average, Kc, lower=1e-5, upper=150)
