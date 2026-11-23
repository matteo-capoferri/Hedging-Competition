import pandas as pd

from backtest.backtesting import price_and_deltas
from backtest.hedge import rebalance_future, target_futures
from models.estimate_rho import estimate_rho


def run_backtest(dates, pricing_arguments, spot, smiles,
                 barrels=1_000_000, lot_size=1_000):
    """Delta hedge dall'emissione, con P&L in USD prima degli interessi.

    La prima data corrisponde all'apertura del collar a costo zero.
    Gli strike sono fissi e la correlazione viene stimata a ogni seduta.
    """
    dates = pd.DatetimeIndex(dates)
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Servono date valide, uniche e in ordine cronologico.")

    contracts = pricing_arguments["contracts"]
    positions = dict.fromkeys(contracts, 0)
    previous_settlements = {}
    previous_value = None
    cash = 0.0
    rows = []

    for today in dates:
        rho = estimate_rho(today, spot, contracts)
        daily_arguments = dict(pricing_arguments, correlation=rho)
        value, delta_april, delta_may = price_and_deltas(
            today, "MonteCarlo", daily_arguments, spot, smiles
        )
        collar_pnl = 0.0 if previous_value is None else barrels * (value - previous_value)

        futures_pnl = 0.0
        execution_pnl = 0.0
        for contract, delta in zip(contracts, (delta_april, delta_may)):
            old_position = positions[contract]
            target = target_futures(delta, barrels, lot_size)
            if old_position == 0 and target == 0:
                continue

            quote = spot.loc[(today, contract)]
            settlement = quote["settlement"]
            if old_position != 0:
                price_change = settlement - previous_settlements[contract]
                futures_pnl += old_position * lot_size * price_change

            positions[contract], trade_pnl = rebalance_future(
                old_position, target, quote["bid"], quote["ask"], settlement, lot_size
            )
            execution_pnl += trade_pnl
            previous_settlements[contract] = settlement

        cash += futures_pnl + execution_pnl
        rows.append({
            "date": today,
            "rho": rho,
            "collar_value_per_barrel": value,
            "delta_april": delta_april,
            "delta_may": delta_may,
            "position_april": positions[contracts[0]],
            "position_may": positions[contracts[1]],
            "collar_pnl": collar_pnl,
            "futures_pnl": futures_pnl,
            "execution_pnl": execution_pnl,
            "daily_pnl": collar_pnl + futures_pnl + execution_pnl,
            "cash": cash,
            "portfolio_value": barrels * value + cash,
        })
        previous_value = value

    return pd.DataFrame(rows).set_index("date")
