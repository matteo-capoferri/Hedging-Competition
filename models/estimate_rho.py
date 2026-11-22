import numpy as np
import pandas as pd


def estimate_rho(today, spot, contracts, window=60, min_obs=20, fallback=0.8):
    """Correlazione dei log-rendimenti dei due futures osservati fino a oggi.

    Usa al massimo window coppie valide. Con dati insufficienti o una
    varianza nulla restituisce fallback (eventualmente l'ultima stima).
    """
    if not 2 <= min_obs <= window:
        raise ValueError("Require 2 <= min_obs <= window")
    if not -1 <= fallback <= 1:
        raise ValueError("Fallback correlation must be between -1 and 1")

    prices = spot["settlement"].unstack().reindex(columns=list(contracts)).sort_index()
    history = prices.loc[:pd.Timestamp(today)]
    valid_prices = history.where((history > 0) & np.isfinite(history))
    returns = np.log(valid_prices).diff().dropna().tail(window)
    if len(returns) < min_obs or (returns.nunique() < 2).any():
        return fallback

    first, second = contracts
    rho = returns[first].corr(returns[second])
    return float(np.clip(rho, -1.0, 1.0)) if np.isfinite(rho) else fallback
