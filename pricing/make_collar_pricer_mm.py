import numpy as np

from models.black76 import black_76
from pricing.select_vols import select_vols


def make_collar_pricer_mm(observed_prices, uses_april, time_steps, april_market,
                          may_market, pricing_arguments):
    """
    INPUT:
        observed_prices = settlements of completed fixings.
        uses_april = contract mask for future fixings.
        time_steps = remaining time intervals in years.
        april_market, may_market = reference price and smile for each future.
        pricing_arguments = fixed strikes Kp/Kc, DF and correlation.

    OUTPUT:
        collar_value = function returning the collar value at two futures prices.

    Match moments of the remaining average; keep past fixings deterministic.
    """
    n_remaining = len(time_steps)
    n_fixings = len(observed_prices) + n_remaining
    remaining_weight = n_remaining / n_fixings
    observed_sum = np.sum(observed_prices)
    residual_put = (n_fixings * pricing_arguments["Kp"] - observed_sum) / n_remaining
    residual_call = (n_fixings * pricing_arguments["Kc"] - observed_sum) / n_remaining
    discount_factor = pricing_arguments["DF"]
    correlation = pricing_arguments.get("correlation", 0.8)
    ref_april, smile_april = april_market
    ref_may, smile_may = may_market

    fixing_times = np.cumsum(time_steps)
    shared_times = np.minimum.outer(fixing_times, fixing_times)
    same_contract = uses_april[:, None] == uses_april[None, :]
    correlations = np.where(same_contract, 1.0, correlation)

    def collar_value(april, may):
        forward_prices = np.where(uses_april, april, may)
        forward_avg = forward_prices.mean()

        def option_value(strike, kind):
            if strike <= 0:
                return 0.0 if kind == "Put" else forward_avg - strike
            vol_april, vol_may = select_vols(
                strike, ref_april, ref_may, forward_avg, smile_april, smile_may,
            )
            volatilities = np.where(uses_april, vol_april, vol_may)
            covariance = np.outer(forward_prices, forward_prices) * np.expm1(
                correlations * np.outer(volatilities, volatilities) * shared_times
            )
            variance = covariance.mean()
            total_vol = np.sqrt(np.log1p(variance / forward_avg**2))
            return black_76(forward_avg, strike, 1.0, total_vol, kind)

        put = option_value(residual_put, "Put")
        call = option_value(residual_call, "Call")
        return discount_factor * remaining_weight * (put - call)

    return collar_value
