import warnings

import numpy as np
import pandas as pd

from models.estimate_rho import estimate_rho
from models.estimate_iv_beta import estimate_iv_beta
from models.estimate_delta_beta import estimate_delta_beta
from models.black76 import implied_volatility, black_greeks
from preprocessing.sofr import discount_factor
from sensitivities.collar_vegas import collar_vegas
from sensitivities.price_and_deltas import price_and_deltas


def prepare_day(today, pricing_arguments, spot, quotes, sofr, payment_date):
    """
    INPUT:
        today = valuation date; pricing_arguments = fixed collar settings.
        spot, quotes = full market histories; quotes may be None.
        sofr = optional funding history; payment_date = collar payment date.
    OUTPUT:
        today_quotes = option quotes for today.
        today_args = pricing settings with today's correlation and discount factor.
        cash_growth = interest factor for yesterday's cash.
    """
    contracts = pricing_arguments["contracts"]
    today_quotes = None if quotes is None else quotes.loc[quotes["date"] == today]
    rho = estimate_rho(today, spot, contracts)
    today_args = dict(pricing_arguments, correlation=rho)
    cash_growth = 1.0
    if sofr is not None:
        rate = sofr.at[today, "known_rate"]
        today_args["DF"] = discount_factor(today, payment_date, rate)
        cash_growth = sofr.at[today, "cash_growth"]

    return today_quotes, today_args, cash_growth


def evaluate_collar(today, method, today_args, spot, smiles):
    """
    INPUT:
        today, method, today_args = valuation date, pricing method and settings.
        spot, smiles = futures prices and daily volatility smiles.
    OUTPUT:
        value, deltas, vegas = collar value and April/May sensitivities per barrel.
    """
    collar_value, delta_april, delta_may = price_and_deltas(
        today, method, today_args, spot, smiles
    )
    vega_april, vega_may = collar_vegas(
        today, method, today_args, spot, smiles
    )

    return collar_value, (delta_april, delta_may), (vega_april, vega_may)


def prepare_options(today, contracts, spot, quotes, chosen_options, prev_option_tickers, delta_weights):
    """
    INPUT:
        today, contracts = valuation date and the two collar futures.
        spot, quotes = market histories; quotes may be None.
        chosen_options = previous selections; prev_option_tickers = actual holdings.
        delta_weights = last valid proxy coefficients, updated when a fit is available.
    OUTPUT:
        chosen_options = new selections by collar leg.
        option_quotes, option_greeks = quotes and Greeks by ticker, including old holdings.
        option_results = selection details for the daily report.
    """
    chosen_options = chosen_options.copy()
    option_results = {}  # Report columns by collar leg: selections, holdings and costs.
    option_quotes = {}   # Full quote rows by ticker, including old holdings for MTM.
    option_greeks = {}   # Today's delta/vega by ticker, reused for actual holdings.
    if quotes is not None:
        for contract, leg in zip(contracts, ("april", "may")):
            excluded = []
            iv_beta, beta_r2 = None, None
            while True:
                chosen_option = select_option(today, contract, quotes, chosen_options[contract], excluded)
                if chosen_option is None:
                    break
                option_underlying = chosen_option["actual_underlying_ticker"]
                iv_beta, beta_r2 = estimate_iv_beta(today, contract, option_underlying, quotes)
                weights = estimate_delta_beta(today, option_underlying, contracts, spot, quotes)
                if iv_beta is not None and weights is not None:
                    delta_weights[option_underlying] = weights
                    break
                # Exclude this underlying from today's selection, not from the history.
                excluded.append(option_underlying)
            chosen_options[contract] = chosen_option
            option_results[f"selected_{leg}_iv_beta"] = iv_beta
            option_results[f"selected_{leg}_beta_r2"] = beta_r2
            for field in ("ticker", "delta", "vega", "quote_date"):
                option_results[f"selected_{leg}_{field}"] = None
            if chosen_option is None:
                continue
            chosen_delta, chosen_vega = evaluate_option(today, chosen_option, spot, quotes)
            option_results.update({f"selected_{leg}_ticker": chosen_option["option_ticker"],
                                   f"selected_{leg}_delta": chosen_delta,
                                   f"selected_{leg}_vega": chosen_vega,
                                   f"selected_{leg}_quote_date": chosen_option["date"]})
            option_quotes[chosen_option["option_ticker"]] = chosen_option
            option_greeks[chosen_option["option_ticker"]] = (chosen_delta, chosen_vega)

        # Cover yesterday's holdings not already valued among today's selections.
        # Their quotes are needed for MTM/closing, and their Greeks if closing fails.
        # Use today's quote when present; otherwise carry the last quote and do not trade.
        for contract in contracts:
            held_ticker = prev_option_tickers[contract]
            if held_ticker is None or held_ticker in option_quotes:
                continue
            history = quotes.loc[(quotes["option_ticker"] == held_ticker)
                                 & (quotes["date"] <= today)].sort_values("date")
            held_quote = history.iloc[-1]
            if held_quote["date"] < today:
                warnings.warn(f"{held_ticker} on {today:%Y-%m-%d}: using quote from {held_quote['date']:%Y-%m-%d}; no trade",
                              RuntimeWarning)
            option_quotes[held_ticker] = held_quote
            option_greeks[held_ticker] = evaluate_option(today, held_quote, spot, quotes)

    return chosen_options, option_quotes, option_greeks, option_results


def select_option(today, contract, quotes, previous=None, excluded=()):
    """
    INPUT:
        today = current market date.
        contract = underlying future to hedge.
        quotes = filtered option quotes with ticker and expiration.
        previous = previously selected row, or None.
        excluded = underlying tickers excluded from today's candidates, not history.
    OUTPUT:
        option = selected row, or None if no suitable option exists.
            date remains the original quote date when carrying prices forward.

    Keep the option while at least 15 days remain and 0.95 <= strike / future <= 1.05.
    New selections require 30 days; rank by ATM distance, then bid-ask spread.
    Try the requested future first, then positive correlations in descending order.
    Correlations use up to 60 recent sessions and at least 20 common log returns.
    """
    daily = quotes.loc[(quotes["date"] == today)
                       & ~quotes["actual_underlying_ticker"].isin(excluded)]
    if previous is not None and previous["actual_underlying_ticker"] in excluded:
        previous = None
    if previous is not None and (previous["expiration"] - today).days >= 15:
        current = daily.loc[daily["option_ticker"] == previous["option_ticker"]]
        alternatives = daily.loc[
            (daily["actual_underlying_ticker"] == previous["actual_underlying_ticker"])
            & (daily["expiration"] == previous["expiration"])
        ]
        still_atm = True
        if not alternatives.empty:
            future_price = alternatives["underlying_settlement"].iloc[0]
            moneyness = previous["strike"] / future_price
            still_atm = 0.95 <= moneyness <= 1.05
        if still_atm:
            if not current.empty:
                return current.iloc[0].copy()
            warnings.warn(
                f"{previous['option_ticker']} on {today:%Y-%m-%d}: using filtered quote "
                f"from {previous['date']:%Y-%m-%d}; no current eligible quote. "
                "ATM status cannot be verified when no same-expiry quotes are available.",
                RuntimeWarning, stacklevel=2,
            )
            return previous.copy()
    # Start with the original underlying. Build the fallback order only if needed.
    underlying_order = [contract]
    for underlying in underlying_order:
        candidates = daily.loc[
            (daily["actual_underlying_ticker"] == underlying)
            & ((daily["expiration"] - today).dt.days >= 30)
        ].copy()
        if not candidates.empty:
            candidates["distance"] = abs(candidates["strike"] - candidates["underlying_settlement"])
            candidates["spread"] = candidates["option_ask"] - candidates["option_bid"]
            return candidates.sort_values(["distance", "spread", "option_ticker"]).iloc[0].copy()

        if underlying == contract:
            history = quotes.loc[quotes["date"] <= today,
                                 ["date", "actual_underlying_ticker", "underlying_settlement"]]
            prices = history.drop_duplicates().pivot(
                index="date", columns="actual_underlying_ticker", values="underlying_settlement"
            ).sort_index()
            if contract not in prices.columns:
                return None
            returns = np.log(prices).diff().tail(60)
            correlations = returns.corr(min_periods=20)[contract].drop(contract)
            correlations = correlations.loc[correlations > 0].sort_values(ascending=False)
            underlying_order.extend(correlations.index.tolist())
    return None


def evaluate_option(today, option, spot, quotes):
    """
    INPUT:
        today = valuation date; option = latest available quote for this option.
        spot, quotes = histories used to find the underlying futures price.
    OUTPUT:
        delta, vega = Greeks per barrel, using remaining time as of today.

    Infer IV from the quote's original date, then calculate today's Greeks.
    """
    option_iv = implied_volatility(
        option["underlying_settlement"], option["strike"],
        option["time_to_expiry_days"] / 365,
        option["settlement_price"], option["option_type"],
    )
    option_underlying = option["actual_underlying_ticker"]
    # Same Greeks in both cases; only the source of the futures price differs.
    # Use spot when available, otherwise the latest underlying price in quotes.
    if (today, option_underlying) in spot.index:
        latest_future_price = spot.at[(today, option_underlying), "settlement"]
    else:
        history = quotes.loc[(quotes["actual_underlying_ticker"] == option_underlying)
                             & (quotes["date"] <= today)].sort_values("date")
        latest_future_price = history["underlying_settlement"].iloc[-1]
    return black_greeks(
        latest_future_price, option["strike"],
        (option["expiration"] - today).days / 365,
        option_iv, option["option_type"],
    )


def settle_day(today, contracts, spot, option_quotes, prev_futures, prev_option_qty,
               prev_option_tickers, prev_future_prices, prev_option_prices,
               prev_cash, cash_growth, collar_value, prev_collar_value,
               execution_pnl, option_execution_pnl, barrels, lot_size):
    """
    INPUT:
        today, contracts, spot, option_quotes = today's date, contracts and marks.
        prev_futures, prev_option_qty, prev_option_tickers = yesterday's holdings.
        prev_future_prices, prev_option_prices = yesterday's settlement prices.
        prev_cash, cash_growth = previous cash balance and its interest factor.
        collar_value, prev_collar_value = today's and yesterday's collar value per barrel.
        execution_pnl, option_execution_pnl = today's futures and option execution costs.
        barrels, lot_size = collar notional and barrels per traded contract.
    OUTPUT:
        cash = updated cash balance.
        pnl = daily P&L components and portfolio value in USD.

    Settle yesterday's quantities; today's trades contribute execution costs only.
    """
    futures_pnl = 0.0
    options_pnl = 0.0
    for contract in contracts:
        if prev_futures[contract] != 0:
            price_change = spot.at[(today, contract), "settlement"] - prev_future_prices[contract]
            futures_pnl += prev_futures[contract] * lot_size * price_change
        old_ticker = prev_option_tickers[contract]
        if old_ticker is not None:
            price_change = option_quotes[old_ticker]["settlement_price"] - prev_option_prices[old_ticker]
            options_pnl += prev_option_qty[contract] * lot_size * price_change

    interest_pnl = prev_cash * (cash_growth - 1)
    collar_pnl = barrels * (collar_value - (0.0 if prev_collar_value is None else prev_collar_value))
    cash = prev_cash + futures_pnl + options_pnl + execution_pnl + option_execution_pnl + interest_pnl
    daily_pnl = collar_pnl + cash - prev_cash

    return cash, {
        "collar_pnl": collar_pnl, "futures_pnl": futures_pnl, "options_pnl": options_pnl,
        "execution_pnl": execution_pnl, "option_execution_pnl": option_execution_pnl,
        "interest_pnl": interest_pnl, "hedged_pnl": daily_pnl,
        "cash": cash, "portfolio_value": barrels * collar_value + cash,
    }


def check_accounting(result):
    """Controlla la riconciliazione giornaliera e restituisce l'errore in USD."""
    # Controllo condiviso da esecuzioni reali e test: non cambia il P&L.
    pnl_from_values = result["portfolio_value"].diff()
    pnl_from_values.iloc[0] = result["portfolio_value"].iloc[0]
    pnl_column = "hedged_pnl" if "hedged_pnl" in result else "delta_hedged_pnl"
    error = result[pnl_column] - pnl_from_values
    if not np.isfinite(error).all() or not np.allclose(error, 0.0, rtol=0.0, atol=1e-6):
        raise ValueError("Il P&L per componenti non riconcilia con il valore del portafoglio.")
    return error
