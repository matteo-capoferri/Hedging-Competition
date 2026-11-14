import numpy as np
from scipy.optimize import brentq

from models.black76 import black_76


def calculate_collar_strike_mm(price_april, price_may, smiles, issue_date, contracts, future,
                               time_steps, call_strike, correlation=0.8):
    """
    INPUT:
        price_april = April futures settlement at issue.
        price_may = May futures settlement at issue.
        smiles = implied volatility smiles indexed by contract and date.
        issue_date = date on which the collar strikes are set.
        contracts = April and May futures tickers, in this order.
        future = fixing calendar with a future_ticker for each date.
        time_steps = consecutive intervals in years, starting at issue.
        call_strike = fixed strike of the collar call.
        correlation = correlation between the two futures shocks.

    OUTPUT:
        put_strike = put strike that matches the call value at issue.
    """
    april, may = contracts
    smile_april = smiles[april][issue_date]
    smile_may = smiles[may][issue_date]

    fixing_times = np.cumsum(time_steps)
    times_april = fixing_times[future["future_ticker"] == april]
    times_may = fixing_times[future["future_ticker"] == may]
    weight_april = len(times_april) / len(fixing_times)
    weight_may = len(times_may) / len(fixing_times)
    forward_avg = weight_april * price_april + weight_may * price_may

    shared_april = np.minimum.outer(times_april, times_april)
    shared_may = np.minimum.outer(times_may, times_may)
    shared_cross = np.minimum.outer(times_april, times_may)

    def option_value(strike, kind):
        if strike == 0:
            return 0.0 if kind == "Put" else forward_avg

        relative_strike = strike / forward_avg
        vol_april = np.interp(price_april * relative_strike, smile_april["K"], smile_april["IV"])
        vol_may = np.interp(price_may * relative_strike, smile_may["K"], smile_may["IV"])

        var_april = price_april**2 * np.expm1(vol_april**2 * shared_april).mean()
        var_may = price_may**2 * np.expm1(vol_may**2 * shared_may).mean()
        covariance = price_april * price_may * np.expm1(
            correlation * vol_april * vol_may * shared_cross
        ).mean()
        variance = (
            weight_april**2 * var_april + weight_may**2 * var_may
            + 2 * weight_april * weight_may * covariance
        )
        total_vol = np.sqrt(np.log1p(variance / forward_avg**2))
        return black_76(F=forward_avg, K=strike, T=1.0, sigma=total_vol, type=kind)

    call_value = option_value(call_strike, "Call")
    return brentq(lambda strike: option_value(strike, "Put") - call_value, 0.0, forward_avg)
