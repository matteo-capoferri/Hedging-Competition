import numpy as np
import pandas as pd
from scipy.stats import linregress

from models.black76 import implied_volatility


def estimate_iv_beta(today, contract, proxy, quotes, window=60, min_obs=20):
    """
    INPUT:
        today = last date allowed in the regression.
        contract, proxy = target and substitute underlying futures.
        quotes = filtered option history, unchanged by candidate exclusions.
        window = maximum number of recent daily IV changes.
        min_obs = minimum number of paired changes required.
    OUTPUT:
        beta, r_squared = slope and fit quality for target IV changes vs proxy IV changes.
            Returns (None, None) when the regression cannot be estimated.

    Interpolate ATM IV between neighbouring strikes, then regress daily changes
    with an intercept. Missing dates stay missing; no forward-filled IVs are used.
    """
    if contract == proxy:
        return 1.0, 1.0
    if not 2 <= min_obs <= window:
        raise ValueError("Require 2 <= min_obs <= window")
    history = quotes.loc[quotes["date"] <= today]
    dates = pd.DatetimeIndex(sorted(history["date"].unique()))[-window-1:]
    history = history.loc[history["date"].isin(dates)
                          & history["actual_underlying_ticker"].isin([contract, proxy])]
    atm = pd.DataFrame(index=dates, columns=[contract, proxy], dtype=float)
    for (day, ticker), daily in history.groupby(["date", "actual_underlying_ticker"]):
        future = daily["underlying_settlement"].iloc[0]
        lower = daily.loc[daily["strike"] <= future, "strike"].max()
        upper = daily.loc[daily["strike"] >= future, "strike"].min()
        if pd.isna(lower) or pd.isna(upper):
            continue  # Do not extrapolate an ATM observation from a distant strike.
        nearby = daily.loc[daily["strike"].isin([lower, upper])]
        values = []
        for row in nearby.itertuples(index=False):
            try:
                iv = implied_volatility(row.underlying_settlement, row.strike,
                                        row.time_to_expiry_days / 365,
                                        row.settlement_price, row.option_type)
                values.append((row.strike, iv))
            except ValueError:
                continue  # An invalid IV is missing data, never a zero volatility.
        if not values:
            continue
        smile = pd.DataFrame(values, columns=["strike", "iv"]).groupby("strike")["iv"].mean()
        if lower in smile.index and upper in smile.index:
            atm.at[day, ticker] = np.interp(future, smile.index, smile.values)

    changes = atm.diff().dropna()
    if len(changes) < min_obs or (changes.var() <= 1e-16).any():
        return None, None
    fit = linregress(changes[proxy], changes[contract])
    if not np.isfinite(fit.slope) or not np.isfinite(fit.rvalue):
        return None, None
    return float(fit.slope), float(fit.rvalue**2)
