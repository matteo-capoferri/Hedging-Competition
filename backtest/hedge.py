from math import isfinite


def target_futures(delta_per_barrel, barrels=1_000_000, lot_size=1_000):
    """Posizione in contratti futures che compensa il delta del collar."""
    exposure = barrels * delta_per_barrel
    return round(-exposure / lot_size)


def rebalance_future(position, target, bid, ask, settlement, lot_size=1_000):
    """Restituisce posizione effettiva e P&L di esecuzione al settlement, in USD."""
    trade = target - position
    if trade == 0:
        return position, 0.0

    execution_price = ask if trade > 0 else bid
    if not isfinite(execution_price):
        return position, 0.0

    execution_pnl = trade * lot_size * (settlement - execution_price)
    return target, execution_pnl


def rebalance_option(position, collar_vega, option_vega, bid, ask, settlement,
                     *, beta, barrels=1_000_000, lot_size=1_000):
    """
    INPUT:
        position = contracts currently held in this specific option.
        collar_vega = collar vega per barrel, per unit decimal volatility.
        option_vega = option vega in the same units.
        bid, ask, settlement = current executable quotes and settlement per barrel.
        beta = 1 for the same underlying; historical IV beta for a substitute.
        barrels = collar notional in barrels.
        lot_size = barrels per option contract.
    OUTPUT:
        position = actual number of contracts after trading.
        execution_pnl = execution P&L relative to settlement, in USD.

    Rebalance one futures-style option: no upfront premium is paid.
    Previous holdings earn their daily settlement P&L separately, before trading.
    """
    if not isfinite(collar_vega) or not isfinite(beta):
        raise ValueError("Collar vega and IV beta must be finite")
    exposure = barrels * collar_vega * beta
    target = 0
    if exposure != 0:
        if not isfinite(option_vega) or option_vega <= 0:
            raise ValueError("A positive option vega is required to hedge the exposure")
        target = round(-exposure / (lot_size * option_vega))

    # Futures-style options use the same execution accounting as futures.
    return rebalance_future(position, target, bid, ask, settlement, lot_size)
