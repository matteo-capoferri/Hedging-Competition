import numpy as np
import pandas as pd

from models.simulate_paths import simulate_paths
from preprocessing.build_calendar import build_calendar
from pricing.price_collar import price_collar
from pricing.select_vols import select_vols


def sens(today, method, pricing_arguments, spot, smiles):
    """Delta di aprile e maggio del collar long put / short call, per unita.

    pricing_arguments richiede Kp, Kc, DF e contracts; gli altri parametri
    hanno i default indicati sotto. La valutazione avviene dopo il fixing
    di oggi. Strike, fixing osservati e DF restano fissi nei bump; le IV
    cambiano secondo sticky moneyness, usando lo strike residuo equivalente.
    """
    if method != "MonteCarlo":
        raise NotImplementedError("sens currently supports only MonteCarlo")

    # 1. Parametri del contratto e della simulazione.
    today = pd.Timestamp(today)
    contracts = pricing_arguments["contracts"]
    put_strike = pricing_arguments["Kp"]
    call_strike = pricing_arguments["Kc"]
    discount_factor = pricing_arguments["DF"]
    eps = pricing_arguments.get("eps", 1e-4)
    n_paths = pricing_arguments.get("n_paths", 100_000)
    correlation = pricing_arguments.get("correlation", 0.8)
    seed = pricing_arguments.get("seed", 42)
    if not 0 < eps < 1:
        raise ValueError("eps must be between zero and one")

    # 2. Fixing gia osservati e calendario residuo.
    observed, future, time_steps = build_calendar(today, contracts)
    observed_prices = np.array([
        spot.at[(row.date, row.future_ticker), "settlement"]
        for row in observed.itertuples(index=False)
    ])
    if not np.all(np.isfinite(observed_prices)):
        raise ValueError("Observed fixings must be finite")
    if future.empty:
        return 0.0, 0.0

    n_remaining = len(future)
    n_fixings = len(observed) + n_remaining
    observed_sum = observed_prices.sum()

    # 3. Prezzi e smile di oggi, prima dei bump.
    april, may = contracts
    uses_april = future["future_ticker"].eq(april).to_numpy()
    has_april_fixings = uses_april.any()
    has_may_fixings = (~uses_april).any()

    # Valori neutri per il future senza fixing residui: non servono sue quote.
    price_april, price_may = 1.0, 1.0
    smile_april = {"K": [1.0], "IV": [0.0]}
    smile_may = {"K": [1.0], "IV": [0.0]}
    if has_april_fixings:
        price_april = spot.at[(today, april), "settlement"]
        smile_april = smiles[april][today]
    if has_may_fixings:
        price_may = spot.at[(today, may), "settlement"]
        smile_may = smiles[may][today]

    # 4. Stessi shock per put, call e tutte le valutazioni bumped.
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_remaining, 2))

    def simulate_leg(strike, bumped_april, bumped_may):
        forward_prices = np.where(uses_april, bumped_april, bumped_may)
        forward_avg = forward_prices.mean()
        effective_strike = (n_fixings * strike - observed_sum) / n_remaining

        if effective_strike <= 0:
            # Put nulla / call lineare: bastano i prezzi attesi.
            return np.broadcast_to(forward_prices, (n_paths, n_remaining))

        # Smile ancorati ai prezzi originali; moneyness aggiornata dal bump.
        vol_april, vol_may = select_vols(
            effective_strike, price_april, price_may, forward_avg,
            smile_april, smile_may,
        )
        paths_april, paths_may = simulate_paths(
            bumped_april, bumped_may, vol_april, vol_may, time_steps,
            n_paths=n_paths, correlation=correlation, shocks=shocks,
        )
        return np.where(uses_april, paths_april, paths_may)

    def collar_value(bumped_april, bumped_may):
        put_paths = simulate_leg(put_strike, bumped_april, bumped_may)
        call_paths = simulate_leg(call_strike, bumped_april, bumped_may)
        return price_collar(
            observed_prices, put_paths, put_strike, call_strike,
            discount_factor, simulated_call_prices=call_paths,
        )

    # 5. Differenze centrali: bump di un solo future alla volta.
    delta_april, delta_may = 0.0, 0.0
    if has_april_fixings:
        bump_april = eps * price_april
        value_up = collar_value(price_april + bump_april, price_may)
        value_down = collar_value(price_april - bump_april, price_may)
        delta_april = (value_up - value_down) / (2 * bump_april)

    if has_may_fixings:
        bump_may = eps * price_may
        value_up = collar_value(price_april, price_may + bump_may)
        value_down = collar_value(price_april, price_may - bump_may)
        delta_may = (value_up - value_down) / (2 * bump_may)

    return delta_april, delta_may
