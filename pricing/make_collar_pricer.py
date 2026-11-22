import numpy as np

from pricing.price_collar import price_collar
from pricing.simulate_leg import simulate_leg


def make_collar_pricer(observed_prices, uses_april, time_steps, april_market,
                       may_market, pricing_arguments):
    """Prepara una valutazione riutilizzabile ai prezzi correnti e nei bump."""
    put_strike = pricing_arguments["Kp"]
    call_strike = pricing_arguments["Kc"]
    discount_factor = pricing_arguments["DF"]
    n_remaining = len(time_steps)
    n_fixings = len(observed_prices) + n_remaining
    observed_sum = observed_prices.sum()
    correlation = pricing_arguments.get("correlation", 0.8)
    n_paths = pricing_arguments.get("n_paths", 100_000)
    seed = pricing_arguments.get("seed", 42)

    # Stessi shock per il prezzo base e per i bump di entrambe le gambe.
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_remaining, 2))

    # Gli strike equivalenti dipendono solo dai fixing già osservati.
    residual_put = (n_fixings * put_strike - observed_sum) / n_remaining
    residual_call = (n_fixings * call_strike - observed_sum) / n_remaining

    def collar_value(april, may):
        """Rivaluta il collar mantenendo fissi gli shock e gli smile di riferimento."""
        put_paths, call_paths = [
            simulate_leg(strike, (april, may), april_market, may_market,
                          uses_april, time_steps, correlation, shocks)
            for strike in (residual_put, residual_call)
        ]
        return price_collar(
            observed_prices, put_paths, put_strike, call_strike,
            discount_factor, simulated_call_prices=call_paths,
        )

    return collar_value
