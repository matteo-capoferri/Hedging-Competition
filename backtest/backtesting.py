import numpy as np
import pandas as pd

from models.estimate_rho import estimate_rho

from preprocessing.build_calendar import build_calendar
from preprocessing.market_inputs import market_inputs
from preprocessing.read_fixings import read_fixings
from pricing.central_delta import central_delta
from pricing.make_collar_pricer import make_collar_pricer
from pricing.make_collar_pricer_mm import make_collar_pricer_mm
from pricing.price_collar import price_collar


def price_and_deltas(today, method, pricing_arguments, spot, smiles):
    """Restituisce il valore del collar per barile e i delta di aprile e maggio.

    pricing_arguments richiede Kp, Kc, DF e contracts; gli altri parametri
    hanno i default indicati sotto. La valutazione avviene dopo il fixing
    di oggi. Strike, fixing osservati e DF restano fissi nei bump; le IV
    cambiano secondo sticky moneyness, usando lo strike residuo equivalente.
    method: "MonteCarlo" or "mm"; both use the same central-difference deltas.
    """
    if method not in ("MonteCarlo", "mm"):
        raise NotImplementedError("Choose MonteCarlo or mm")

    today = pd.Timestamp(today)
    contracts = pricing_arguments["contracts"]
    put_strike = pricing_arguments["Kp"]
    call_strike = pricing_arguments["Kc"]
    discount_factor = pricing_arguments["DF"]
    eps = pricing_arguments.get("eps", 1e-4)
    if not 0 < eps < 1:
        raise ValueError("eps must be between zero and one")

    observed, future, time_steps = build_calendar(today, contracts)
    observed_prices = read_fixings(observed, spot)
    if future.empty:
        value = price_collar(
            observed_prices, np.empty((1, 0)), put_strike, call_strike, discount_factor
        )
        return value, 0.0, 0.0

    april, may = contracts
    uses_april = future["future_ticker"].eq(april).to_numpy()
    has_april_fixings = uses_april.any()
    has_may_fixings = (~uses_april).any()

    april_market = market_inputs(today, april, has_april_fixings, spot, smiles)
    may_market = market_inputs(today, may, has_may_fixings, spot, smiles)
    price_april, price_may = april_market[0], may_market[0]

    if "correlation" not in pricing_arguments:
        rho = estimate_rho(today, spot, contracts)
        pricing_arguments = dict(pricing_arguments, correlation=rho)

    make_pricer = make_collar_pricer_mm if method == "mm" else make_collar_pricer
    collar_value = make_pricer(
        observed_prices, uses_april, time_steps, april_market, may_market, pricing_arguments
    )

    delta_april, delta_may = 0.0, 0.0
    if has_april_fixings:
        delta_april = central_delta(lambda price: collar_value(price, price_may), price_april, eps)
    if has_may_fixings:
        delta_may = central_delta(lambda price: collar_value(price_april, price), price_may, eps)

    value = collar_value(price_april, price_may)
    return value, delta_april, delta_may
