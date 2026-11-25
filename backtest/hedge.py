import warnings

from models.estimate_delta_beta import estimate_delta_beta

from math import isfinite


def rebalance_options(today, contracts, today_quotes, chosen_options, held_option_tickers,
                      held_option_qty, vegas, option_results, barrels, lot_size):
    """
    INPUT:
        today, contracts, today_quotes = date, collar futures and executable quotes.
        chosen_options = desired options; held_option_tickers, held_option_qty = old holdings.
        vegas = April/May collar vegas; option_results = selection details, updated with trades.
        barrels, lot_size = collar notional and barrels per contract.
    OUTPUT:
        held_option_tickers, held_option_qty = actual holdings after trading.
        execution_pnl = total option execution P&L in USD.

    Close a changed ticker before opening its replacement; keep holdings if closing fails.
    """
    held_option_qty = held_option_qty.copy()
    held_option_tickers = held_option_tickers.copy()
    option_order_pnls = []
    if today_quotes is not None:
        for contract, leg, collar_vega in zip(contracts, ("april", "may"), vegas):
            option_results[f"option_{leg}_execution_pnl"] = 0.0
            option_results[f"option_{leg}_status"] = "unchanged"
            chosen_option = chosen_options[contract]
            chosen_ticker = None if chosen_option is None else chosen_option["option_ticker"]
            held_ticker = held_option_tickers[contract]

            beta = option_results[f"selected_{leg}_iv_beta"]

            # Different ticker (including None): close the previous position first.
            if held_ticker is not None and held_ticker != chosen_ticker:
                closing_quote = today_quotes.loc[today_quotes["option_ticker"] == held_ticker]
                if closing_quote.empty:
                    # Keep the old position and skip opening its replacement today.
                    option_results[f"option_{leg}_status"] = "missing_closing_quote"
                    continue
                closing_quote = closing_quote.iloc[0]
                option_qty_after, order_pnl = rebalance_future(
                    held_option_qty[contract], 0, closing_quote["option_bid"],
                    closing_quote["option_ask"], closing_quote["settlement_price"], lot_size,
                )
                held_option_qty[contract] = option_qty_after
                option_results[f"option_{leg}_execution_pnl"] += order_pnl
                option_order_pnls.append(order_pnl)
                if option_qty_after != 0:
                    # Execution was skipped (missing bid/ask); the old hedge is still held.
                    option_results[f"option_{leg}_status"] = "missing_closing_quote"
                    continue
                held_option_tickers[contract] = None
                option_results[f"option_{leg}_status"] = "closed"

            if chosen_option is None:
                # No opening order. Any previous position was handled by the closing block.
                continue
            if chosen_option["date"] != today:
                # An old quote may value a holding, but cannot execute a trade today.
                option_results[f"option_{leg}_status"] = "stale_quote"
                continue
            # Same ticker: trade the difference. After a closure, quantity is zero.
            option_qty_after, order_pnl = rebalance_option(
                held_option_qty[contract], collar_vega, option_results[f"selected_{leg}_vega"],
                chosen_option["option_bid"], chosen_option["option_ask"], chosen_option["settlement_price"],
                beta=beta, barrels=barrels, lot_size=lot_size,
            )
            held_option_qty[contract] = option_qty_after
            held_option_tickers[contract] = chosen_ticker if option_qty_after != 0 else None
            option_results[f"option_{leg}_execution_pnl"] += order_pnl
            option_order_pnls.append(order_pnl)
            option_results[f"option_{leg}_status"] = "rebalanced"

        for contract, leg in zip(contracts, ("april", "may")):
            option_results[f"held_{leg}_ticker"] = held_option_tickers[contract]
            option_results[f"held_{leg}_quantity"] = held_option_qty[contract]

    return held_option_tickers, held_option_qty, sum(option_order_pnls)


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


def rebalance_futures(today, contracts, spot, quotes, held_future_qty, held_option_tickers,
                      held_option_qty, option_quotes, option_greeks, deltas, delta_weights,
                      option_results, barrels, lot_size):
    """
    INPUT:
        today, contracts, spot, quotes = date, hedge futures and market histories.
        held_future_qty = old futures holdings; held_option_tickers, held_option_qty = new option holdings.
        option_quotes, option_greeks = marks and Greeks for actual options.
        deltas = collar deltas; delta_weights = proxy mappings, updated with valid estimates.
        option_results = report details, updated with option delta exposures.
        barrels, lot_size = collar notional and barrels per contract.
    OUTPUT:
        held_future_qty = actual new futures holdings.
        execution_pnl = total futures execution P&L in USD.

    Cover the collar and the options actually held, including substitute underlyings.
    """
    held_future_qty = held_future_qty.copy()
    future_order_pnls = []
    # Map every actual option holding to both tradable futures, including substitutes.
    option_deltas = dict.fromkeys(contracts, 0.0)
    for contract in contracts:
        ticker = held_option_tickers[contract]
        if ticker is None:
            continue
        underlying = option_quotes[ticker]["actual_underlying_ticker"]
        weights = estimate_delta_beta(today, underlying, contracts, spot, quotes)
        if weights is None:
            warnings.warn(f"{underlying} on {today:%Y-%m-%d}: keeping last valid delta weights",
                          RuntimeWarning)
            weights = delta_weights[underlying]
        delta_weights[underlying] = weights
        exposure = held_option_qty[contract] * lot_size * option_greeks[ticker][0]
        for future, weight in zip(contracts, weights):
            option_deltas[future] += exposure * weight

    # Futures cover the collar plus the options actually held after trading.
    for contract, leg, collar_delta in zip(contracts, ("april", "may"), deltas):
        # Both exposures are in barrels; the target is in futures contracts.
        option_delta_total = option_deltas[contract]
        portfolio_delta = barrels * collar_delta + option_delta_total
        option_results[f"option_{leg}_delta_exposure"] = option_delta_total
        future_qty_before = held_future_qty[contract]
        target_future_qty = target_futures(portfolio_delta, 1, lot_size)
        if future_qty_before == 0 and target_future_qty == 0:
            # No position and no order: no quote lookup is needed for this future.
            continue

        future_quote = spot.loc[(today, contract)]
        today_settlement = future_quote["settlement"]

        held_future_qty[contract], order_pnl = rebalance_future(
            future_qty_before, target_future_qty, future_quote["bid"], future_quote["ask"], today_settlement, lot_size
        )
        future_order_pnls.append(order_pnl)

    return held_future_qty, sum(future_order_pnls)
def target_futures(delta_per_barrel, barrels=1_000_000, lot_size=1_000):
    """Posizione in contratti futures che compensa il delta del collar."""
    exposure = barrels * delta_per_barrel
    return round(-exposure / lot_size)


