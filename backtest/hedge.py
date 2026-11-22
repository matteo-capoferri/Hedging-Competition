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
