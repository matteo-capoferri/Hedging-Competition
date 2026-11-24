import pandas as pd
import warnings

from backtest.backtesting import price_and_deltas
from backtest.check_accounting import check_accounting
from backtest.hedge import rebalance_future, rebalance_option, target_futures
from backtest.select_option import select_option
from models.black76 import implied_volatility, black_greeks
from models.estimate_rho import estimate_rho
from preprocessing.sofr import discount_factor
from pricing.collar_vegas import collar_vegas


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
        lot_size = number of barrels represented by one futures contract.
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
            Selected option tickers, quote dates and Greeks when quotes is provided.
            Option Greeks are per barrel; vega is per unit decimal volatility.

    Reprice the collar and rebalance its futures delta hedge each day.
    Strikes remain fixed; futures correlation is estimated again each day.
    Initial wealth is zero, so the initial collar value enters P&L. The final
    payoff stays in the collar value and is not also credited to cash.
    Step 5 hedges the delta of the collar and actual option holdings together.
    Option settlements and execution P&L are not yet included in cash or total P&L.
    Trading options on substitute underlying futures is disabled for now.
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
    prev_collar_value = None
    cash = 0.0
    daily_results = []

    for today in dates:
        # --- 1. Prepare market inputs and mark yesterday's portfolio to market ---
        rho = estimate_rho(today, spot, contracts)
        today_args = dict(pricing_arguments, correlation=rho)
        interest_pnl = 0.0
        if sofr is not None:
            rate = sofr.at[today, "known_rate"]
            today_args["DF"] = discount_factor(today, payment_date, rate)
            interest_pnl = cash * (sofr.at[today, "cash_growth"] - 1)
        futures_pnl = 0.0
        for contract in contracts:
            if held_future_qty[contract] != 0:
                today_settlement = spot.at[(today, contract), "settlement"]
                price_change = today_settlement - prev_future_prices[contract]
                futures_pnl += held_future_qty[contract] * lot_size * price_change

        # The existing pricer returns value and deltas together; call it only once.
        collar_value, delta_april, delta_may = price_and_deltas(
            today, method, today_args, spot, smiles
        )
        collar_pnl = barrels * (collar_value - (0.0 if prev_collar_value is None else prev_collar_value))
        # TODO: Add daily settlement P&L for yesterday's option holdings.

        # --- 2. Complete today's collar sensitivities (deltas are already available) ---
        vega_april, vega_may = collar_vegas(
            today, method, today_args, spot, smiles
        )

        # --- 3. Select hedge options and calculate their Greeks ---
        option_results = {}
        if quotes is not None:
            for contract, leg in zip(contracts, ("april", "may")):
                chosen_option = select_option(today, contract, quotes, chosen_options[contract])
                chosen_options[contract] = chosen_option
                for field in ("ticker", "delta", "vega", "quote_date"):
                    option_results[f"selected_{leg}_{field}"] = None
                if chosen_option is None:
                    continue
                option_iv = implied_volatility(
                    chosen_option["underlying_settlement"], chosen_option["strike"],
                    chosen_option["time_to_expiry_days"] / 365,
                    chosen_option["settlement_price"], chosen_option["option_type"],
                )
                option_underlying = chosen_option["actual_underlying_ticker"]
                if (today, option_underlying) in spot.index:
                    latest_future_price = spot.at[(today, option_underlying), "settlement"]
                else:
                    history = quotes.loc[(quotes["actual_underlying_ticker"] == option_underlying)
                                         & (quotes["date"] <= today)].sort_values("date")
                    latest_future_price = history["underlying_settlement"].iloc[-1]
                chosen_delta, chosen_vega = black_greeks(
                    latest_future_price, chosen_option["strike"],
                    (chosen_option["expiration"] - today).days / 365,
                    option_iv, chosen_option["option_type"],
                )
                option_results.update({f"selected_{leg}_ticker": chosen_option["option_ticker"],
                                       f"selected_{leg}_delta": chosen_delta,
                                       f"selected_{leg}_vega": chosen_vega,
                                       f"selected_{leg}_quote_date": chosen_option["date"]})

        # --- 4. Same ticker: adjust quantity. Different ticker: close, then open. ---
        if quotes is not None:
            today_quotes = quotes.loc[quotes["date"] == today]
            for contract, leg, collar_vega in zip(contracts, ("april", "may"), (vega_april, vega_may)):
                option_results[f"option_{leg}_execution_pnl"] = 0.0
                option_results[f"option_{leg}_status"] = "unchanged"
                chosen_option = chosen_options[contract]
                chosen_ticker = None if chosen_option is None else chosen_option["option_ticker"]
                held_ticker = held_option_tickers[contract]

                beta = 1.0
                if chosen_option is not None and chosen_option["actual_underlying_ticker"] != contract:
                    option_results[f"option_{leg}_status"] = "proxy_hedge_not_implemented"
                    continue

                # Different ticker (including None): close the previous position first.
                if held_ticker is not None and held_ticker != chosen_ticker:
                    closing_quote = today_quotes.loc[today_quotes["option_ticker"] == held_ticker]
                    if closing_quote.empty:
                        option_results[f"option_{leg}_status"] = "missing_closing_quote"
                        continue
                    closing_quote = closing_quote.iloc[0]
                    option_qty_after, order_pnl = rebalance_future(
                        held_option_qty[contract], 0, closing_quote["option_bid"],
                        closing_quote["option_ask"], closing_quote["settlement_price"], lot_size,
                    )
                    held_option_qty[contract] = option_qty_after
                    option_results[f"option_{leg}_execution_pnl"] += order_pnl
                    if option_qty_after != 0:
                        option_results[f"option_{leg}_status"] = "missing_closing_quote"
                        continue
                    held_option_tickers[contract] = None
                    option_results[f"option_{leg}_status"] = "closed"

                if chosen_option is None:
                    continue
                if chosen_option["date"] != today:
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
                option_results[f"option_{leg}_status"] = "rebalanced"

            for contract, leg in zip(contracts, ("april", "may")):
                option_results[f"held_{leg}_ticker"] = held_option_tickers[contract]
                option_results[f"held_{leg}_quantity"] = held_option_qty[contract]

        # --- 5. Hedge the delta of the collar and the options actually held ---
        execution_pnl = 0.0
        for contract, leg, collar_delta in zip(contracts, ("april", "may"), (delta_april, delta_may)):
            held_option_delta = 0.0
            held_ticker = held_option_tickers[contract]
            if held_ticker is not None:
                if held_ticker == option_results[f"selected_{leg}_ticker"]:
                    held_option_delta = option_results[f"selected_{leg}_delta"]
                else:
                    # A failed replacement leaves the old option in the portfolio.
                    history = quotes.loc[(quotes["option_ticker"] == held_ticker)
                                         & (quotes["date"] <= today)].sort_values("date")
                    held_quote = history.iloc[-1]
                    if held_quote["date"] < today:
                        warnings.warn(f"{held_ticker} on {today:%Y-%m-%d}: IV from {held_quote['date']:%Y-%m-%d}; delta uses today's future",
                                      RuntimeWarning)
                    option_iv = implied_volatility(
                        held_quote["underlying_settlement"], held_quote["strike"],
                        held_quote["time_to_expiry_days"] / 365, held_quote["settlement_price"], held_quote["option_type"],
                    )
                    held_option_delta, _ = black_greeks(
                        spot.at[(today, contract), "settlement"], held_quote["strike"],
                        (held_quote["expiration"] - today).days / 365, option_iv, held_quote["option_type"],
                    )
            # Both exposures are in barrels; the target is in futures contracts.
            option_delta_total = held_option_qty[contract] * lot_size * held_option_delta
            portfolio_delta = barrels * collar_delta + option_delta_total
            option_results[f"option_{leg}_delta_exposure"] = option_delta_total
            future_qty_before = held_future_qty[contract]
            target_future_qty = target_futures(portfolio_delta, 1, lot_size)
            if future_qty_before == 0 and target_future_qty == 0:
                continue

            future_quote = spot.loc[(today, contract)]
            today_settlement = future_quote["settlement"]

            held_future_qty[contract], order_pnl = rebalance_future(
                future_qty_before, target_future_qty, future_quote["bid"], future_quote["ask"], today_settlement, lot_size
            )
            execution_pnl += order_pnl
            prev_future_prices[contract] = today_settlement

        # --- 6. Update cash, portfolio value and daily P&L ---
        # TODO: Include futures-style option daily settlements in cash, without double counting.
        cash += futures_pnl + execution_pnl + interest_pnl
        daily_results.append({
            **option_results,
            "date": today,
            "rho": rho,
            "discount_factor": today_args["DF"],
            "interest_pnl": interest_pnl,
            "collar_value_per_barrel": collar_value,
            "delta_april": delta_april,
            "delta_may": delta_may,
            "vega_april_per_barrel": vega_april,
            "vega_may_per_barrel": vega_may,
            "april_vol_up_1bp_pnl_usd": barrels * vega_april * 0.0001,
            "may_vol_up_1bp_pnl_usd": barrels * vega_may * 0.0001,
            "position_april": held_future_qty[contracts[0]],
            "position_may": held_future_qty[contracts[1]],
            "collar_pnl": collar_pnl,
            "futures_pnl": futures_pnl,
            "execution_pnl": execution_pnl,
            "delta_hedged_pnl": collar_pnl + futures_pnl + execution_pnl + interest_pnl,
            "cash": cash,
            "portfolio_value": barrels * collar_value + cash,
        })
        prev_collar_value = collar_value

    # --- Assemble results and reconcile P&L with changes in portfolio value ---
    result = pd.DataFrame(daily_results).set_index("date")
    # Monitora la contabilità anche sui dati reali; verificato nei test contabili.
    result["accounting_error"] = check_accounting(result)
    return result
