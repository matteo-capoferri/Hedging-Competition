import numpy as np
import pandas as pd


def estimate_delta_beta(today, underlying, contracts, spot, quotes, window=60, min_obs=20):
    """
    INPUT:
        today = last date allowed in the regression.
        underlying = future underlying the hedge option.
        contracts = April and May futures used for delta hedging.
        spot, quotes = futures and option price histories.
        window, min_obs = maximum and minimum number of daily price changes.
    OUTPUT:
        weights = delta exposure to each hedge future, or None if data is insufficient.

    Regress changes in the substitute future on changes in April and May prices.
    """
    if underlying in contracts:
        return tuple(float(contract == underlying) for contract in contracts)
    prices = spot["settlement"].unstack().reindex(columns=list(contracts))
    history = quotes.loc[(quotes["date"] <= today)
                         & (quotes["actual_underlying_ticker"] == underlying)]
    proxy = history.groupby("date")["underlying_settlement"].first()
    prices = prices.join(proxy.rename("proxy"))
    changes = prices.loc[prices.index <= today].sort_index().diff().tail(window).dropna()
    if len(changes) < min_obs:
        return None
    x = np.column_stack([np.ones(len(changes)), changes[list(contracts)].to_numpy()])
    y = changes["proxy"].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        return None
    return tuple(coefficients[1:])
