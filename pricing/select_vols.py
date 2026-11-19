import numpy as np


def select_vols(strike, price_april, price_may, forward_avg,
                smile_april, smile_may):
    """Return April and May IVs at the basket option's relative strike.

    At issue, forward_avg is the expected fixing average and strike is Kp
    or Kc. During averaging use the residual average's forward and effective
    strike (N * K - observed_sum) / n_remaining, when positive.

    Each smile contains increasing strikes in 'K' and volatilities in 'IV',
    as returned by extract_smile. Outside its strike range, np.interp uses
    the nearest endpoint volatility (the existing pricing convention).

    For sticky moneyness, price_april and price_may are the UNBUMPED smile
    reference futures, while forward_avg is recalculated after each bump.
    Thus interpolation is equivalent to using fixed K / F_reference nodes.
    A non-positive effective strike gives a zero put / linear call under
    positive futures and should be handled directly by the caller.
    """
    if not np.isfinite(strike) or strike < 0:
        raise ValueError("strike must be finite and non-negative")
    prices = np.asarray([price_april, price_may, forward_avg], dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Futures prices and forward_avg must be finite and positive")

    relative_strike = strike / forward_avg
    vol_april = np.interp(
        price_april * relative_strike, smile_april["K"], smile_april["IV"]
    )
    vol_may = np.interp(
        price_may * relative_strike, smile_may["K"], smile_may["IV"]
    )
    return float(vol_april), float(vol_may)
