import numpy as np
import pandas as pd

from models.estimate_rho import estimate_rho
from preprocessing.build_calendar import build_calendar
from preprocessing.market_inputs import market_inputs
from preprocessing.read_fixings import read_fixings
from pricing.make_collar_pricer import make_collar_pricer
from pricing.make_collar_pricer_mm import make_collar_pricer_mm


def collar_vegas(today, method, pricing_arguments, spot, smiles, bump=0.0001):
    """
    INPUT:
        today = valuation date, after today's fixing.
        method = "mm" or "MonteCarlo".
        pricing_arguments = fixed strikes, discount factor and model settings.
        spot, smiles = futures history and daily implied volatility smiles.
        bump = parallel increase in decimal volatility; 0.0001 is one basis point.

    OUTPUT:
        vega_april, vega_may = collar price change per barrel per unit volatility.
        Multiply each vega by bump to obtain the corresponding bump P&L per barrel.

    Shift one smile at a time, keeping futures, strikes and observed fixings fixed.
    """
    if method not in ("MonteCarlo", "mm"):
        raise NotImplementedError("Choose MonteCarlo or mm")
    if not np.isfinite(bump) or bump <= 0:
        raise ValueError("The volatility bump must be positive and finite")

    today = pd.Timestamp(today)
    contracts = pricing_arguments["contracts"]
    observed, future, time_steps = build_calendar(today, contracts)
    if future.empty:
        return 0.0, 0.0

    observed_prices = read_fixings(observed, spot)
    uses_april = future["future_ticker"].eq(contracts[0]).to_numpy()
    active = [uses_april.any(), (~uses_april).any()]
    april_market = market_inputs(today, contracts[0], active[0], spot, smiles)
    may_market = market_inputs(today, contracts[1], active[1], spot, smiles)
    arguments = dict(pricing_arguments)
    if "correlation" not in arguments:
        arguments["correlation"] = estimate_rho(today, spot, contracts)
    if arguments.get("seed", 42) is None:
        arguments["seed"] = int(np.random.default_rng().integers(0, 2**32))

    make_pricer = make_collar_pricer_mm if method == "mm" else make_collar_pricer
    base_pricer = make_pricer(
        observed_prices, uses_april, time_steps, april_market, may_market, arguments
    )
    price_april, price_may = april_market[0], may_market[0]
    base_value = base_pricer(price_april, price_may)
    vegas = [0.0, 0.0]

    for index in range(2):
        if not active[index]:
            continue
        markets = [april_market, may_market]
        price, smile = markets[index]
        shifted_smile = {
            "K": smile["K"],
            "IV": np.asarray(smile["IV"]) + bump,
        }
        markets[index] = (price, shifted_smile)
        shifted_pricer = make_pricer(
            observed_prices, uses_april, time_steps, markets[0], markets[1], arguments
        )
        shifted_value = shifted_pricer(price_april, price_may)
        vegas[index] = (shifted_value - base_value) / bump

    return vegas[0], vegas[1]
