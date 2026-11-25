import pandas as pd

from backtest.backtesting import (
    prepare_day, evaluate_collar, prepare_options, settle_day, check_accounting,
)
from backtest.hedge import rebalance_options, rebalance_futures


def run_backtest(dates, pricing_arguments, spot, smiles,
                 barrels=1_000_000, lot_size=1_000, *, sofr=None, payment_date=None,
                 method="MonteCarlo", quotes=None):
    """
    INPUT:
        dates = unique trading dates in chronological order, starting at issue.
        pricing_arguments = dictionary containing Kp, Kc and contracts (April, May),
            plus optional pricing settings. Supply DF when sofr is None.
        spot = futures settlement, bid and ask prices, indexed by date and ticker.
        smiles = daily volatility smiles: smiles[contract][date] contains K and IV.
        barrels = collar notional in barrels.
        lot_size = barrels per futures or option contract (same size assumed).
        sofr = optional DataFrame on the same dates, containing known_rate for
            discounting and cash_growth for interest on the previous cash balance.
        payment_date = collar payment date; required when sofr is provided.
        method = "MonteCarlo" for simulation or "mm" for moment matching.
        quotes = optional filtered option quotes, including ticker and expiration,
            used to select hedge options and calculate their delta and vega.

    OUTPUT:
        result = DataFrame indexed by trading date, containing:
            Collar value per barrel, deltas, vegas and +1 bp volatility stresses.
            Futures positions in contracts and daily P&L components in USD.
            Cash balance, portfolio value and accounting reconciliation error.
            hedged_pnl = total daily P&L, including option settlements and costs.
            options_pnl = MTM of yesterday's options; option_execution_pnl = today's executions.
            Selected option tickers, quote dates and Greeks when quotes is provided.
            Option Greeks are per barrel; vega is per unit decimal volatility.

    Reprice the collar and rebalance its futures delta hedge each day.
    Strikes remain fixed; futures correlation is estimated again each day.
    Initial wealth is zero, so the initial collar value enters P&L. The final
    payoff stays in the collar value and is not also credited to cash.
    Option settlement changes and execution P&L enter cash daily (futures-style).
    No upfront premium or additional option asset value is counted.
    Substitute options use historical IV beta for sizing and price betas for delta.
    """
    # --- Validate the calendar and funding data ---
    dates = pd.DatetimeIndex(dates)
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Servono date valide, uniche e in ordine cronologico.")
    if sofr is not None:
        if not sofr.index.equals(dates):
            raise ValueError("Preparare il SOFR sulle stesse date del backtest.")
        if payment_date is None or payment_date < dates[-1]:
            raise ValueError("La data di pagamento deve coprire tutto il backtest.")

    # --- Initialize positions, cash and daily results ---
    # prev_* = previous valuation; held_* = actual holdings, updated after trades.
    # chosen_* = selection; target_* = desired quantity; *_after = actual new quantity.
    contracts = pricing_arguments["contracts"]
    held_future_qty = dict.fromkeys(contracts, 0)
    chosen_options = dict.fromkeys(contracts)
    held_option_tickers = dict.fromkeys(contracts)
    held_option_qty = dict.fromkeys(contracts, 0)
    prev_future_prices = {}
    prev_option_prices = {}
    delta_weights = {}  # Last valid mapping from each option underlying to April/May.
    prev_collar_value = None
    cash = 0.0
    daily_results = []

    for today in dates:
        # --- 1. Prepare today's data and preserve yesterday's portfolio ---
        prev_cash = cash
        prev_futures = held_future_qty.copy()
        prev_option_qty = held_option_qty.copy()
        prev_option_tickers = held_option_tickers.copy()
        today_quotes, today_args, cash_growth = prepare_day(
            today, pricing_arguments, spot, quotes, sofr, payment_date
        )

        # --- 2. Price the collar and calculate its sensitivities ---
        collar_value, deltas, vegas = evaluate_collar(today, method, today_args, spot, smiles)

        # --- 3. Select options and value both new and existing holdings ---
        chosen_options, option_quotes, option_greeks, option_results = prepare_options(
            today, contracts, spot, quotes, chosen_options, prev_option_tickers, delta_weights
        )

        # --- 4. Rebalance options, then cover the actual portfolio delta ---
        held_option_tickers, held_option_qty, option_execution_pnl = rebalance_options(
            today, contracts, today_quotes, chosen_options, held_option_tickers,
            held_option_qty, vegas, option_results, barrels, lot_size
        )
        held_future_qty, execution_pnl = rebalance_futures(
            today, contracts, spot, quotes, held_future_qty, held_option_tickers,
            held_option_qty, option_quotes, option_greeks, deltas, delta_weights,
            option_results, barrels, lot_size
        )

        # --- 5. Settle yesterday's holdings and update cash ---
        cash, pnl = settle_day(
            today, contracts, spot, option_quotes, prev_futures, prev_option_qty,
            prev_option_tickers, prev_future_prices, prev_option_prices,
            prev_cash, cash_growth, collar_value, prev_collar_value,
            execution_pnl, option_execution_pnl, barrels, lot_size
        )

        daily_results.append({
            **option_results,
            "date": today,
            "rho": today_args["correlation"],
            "discount_factor": today_args["DF"],
            "collar_value_per_barrel": collar_value,
            "delta_april": deltas[0],
            "delta_may": deltas[1],
            "vega_april_per_barrel": vegas[0],
            "vega_may_per_barrel": vegas[1],
            "april_vol_up_1bp_pnl_usd": barrels * vegas[0] * 0.0001,
            "may_vol_up_1bp_pnl_usd": barrels * vegas[1] * 0.0001,
            "position_april": held_future_qty[contracts[0]],
            "position_may": held_future_qty[contracts[1]],
            **pnl,
            "delta_vega_hedged_pnl" if quotes is not None else "delta_hedged_pnl": pnl["hedged_pnl"],
        })
        # Today's marks become the reference for tomorrow's held positions.
        for contract in contracts:
            if held_future_qty[contract] != 0:
                prev_future_prices[contract] = spot.at[(today, contract), "settlement"]
        prev_option_prices = {
            ticker: option_quotes[ticker]["settlement_price"]
            for ticker in held_option_tickers.values() if ticker is not None
        }
        prev_collar_value = collar_value

    # --- Assemble results and reconcile P&L with changes in portfolio value ---
    result = pd.DataFrame(daily_results).set_index("date")
    # Check that cash flows reconcile with portfolio value changes.
    result["accounting_error"] = check_accounting(result)
    return result
