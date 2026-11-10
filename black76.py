"""Prezzi e volatilità implicite Black-76, senza sconto."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr


def black76(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    option_type: str,
) -> float:
    """Prezzo Black-76 non scontato, con tempo ACT/365F."""
    if option_type not in {"Call", "Put"}:
        raise ValueError(f"Tipo opzione non riconosciuto: {option_type}")
    if not np.isfinite([forward, strike, maturity, sigma]).all():
        raise ValueError("Gli input Black devono essere finiti")
    if forward <= 0 or strike <= 0 or maturity < 0 or sigma < 0:
        raise ValueError("Black richiede F,K positivi e T,sigma non negativi")
    option_sign = 1 if option_type == "Call" else -1
    if maturity == 0 or sigma == 0:
        return max(option_sign * (forward - strike), 0.0)
    v = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strike) / v + v / 2
    d2 = d1 - v
    return float(
        option_sign
        * (forward * ndtr(option_sign * d1) - strike * ndtr(option_sign * d2))
    )


def black_76(F, K, T, sigma, type):
    """Interfaccia del prototipo originale."""
    return black76(F, K, T, sigma, type)


def implied_volatility(forward, strike, maturity, option_type, market_price,
                       *, lower=1e-8, upper=10.0):
    def difference(volatility):
        return black76(forward, strike, maturity, volatility, option_type) - market_price

    return brentq(difference, lower, upper, xtol=1e-10)


def implied_vol(row: pd.Series) -> float:
    """Restituisce NaN se il mid non ammette una radice nell'intervallo cercato."""
    try:
        return implied_volatility(
            row.underlying_settlement, row.strike, row.ttm, row.option_type, row.option_mid,
        )
    except ValueError:
        return np.nan
