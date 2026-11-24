import pandas as pd

from backtest.backtesting import price_and_deltas
from backtest.check_accounting import check_accounting
from backtest.hedge import rebalance_future, target_futures
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
    Options are only selected and analysed here: option trades and their P&L
    are not implemented yet.
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
    contracts = pricing_arguments["contracts"]
    positions = dict.fromkeys(contracts, 0)
    selected_options = dict.fromkeys(contracts)
    previous_settlements = {}
    previous_value = None
    cash = 0.0
    rows = []

    for today in dates:
        # --- 1. Prepare today's market inputs and interest on yesterday's cash ---
        # Existing futures are marked to market in step 5, before trading.
        # TODO: Mark existing option positions using today's available quotes.
        rho = estimate_rho(today, spot, contracts)
        daily_arguments = dict(pricing_arguments, correlation=rho)
        interest_pnl = 0.0
        if sofr is not None:
            rate = sofr.at[today, "known_rate"]
            daily_arguments["DF"] = discount_factor(today, payment_date, rate)
            interest_pnl = cash * (sofr.at[today, "cash_growth"] - 1)
        # --- 2. Value the collar and calculate its delta and vega ---
        value, delta_april, delta_may = price_and_deltas(
            today, method, daily_arguments, spot, smiles
        )
        vega_april, vega_may = collar_vegas(
            today, method, daily_arguments, spot, smiles
        )
        collar_pnl = barrels * (value - (0.0 if previous_value is None else previous_value))

        # --- 3. Select hedge options and calculate their Greeks ---
        option_results = {}
        if quotes is not None:
            for contract, leg in zip(contracts, ("april", "may")):
                option = select_option(today, contract, quotes, selected_options[contract])
                selected_options[contract] = option
                for field in ("ticker", "delta", "vega", "quote_date"):
                    option_results[f"selected_{leg}_{field}"] = None
                if option is None:
                    continue
                volatility = implied_volatility(
                    option["underlying_settlement"], option["strike"],
                    option["time_to_expiry_days"] / 365,
                    option["settlement_price"], option["option_type"],
                )
                underlying = option["actual_underlying_ticker"]
                if (today, underlying) in spot.index:
                    future_price = spot.at[(today, underlying), "settlement"]
                else:
                    history = quotes.loc[(quotes["actual_underlying_ticker"] == underlying)
                                         & (quotes["date"] <= today)].sort_values("date")
                    future_price = history["underlying_settlement"].iloc[-1]
                delta, vega = black_greeks(
                    future_price, option["strike"],
                    (option["expiration"] - today).days / 365,
                    volatility, option["option_type"],
                )
                option_results.update({f"selected_{leg}_ticker": option["option_ticker"],
                                       f"selected_{leg}_delta": delta,
                                       f"selected_{leg}_vega": vega,
                                       f"selected_{leg}_quote_date": option["date"]})

        # --- 4. Rebalance options to hedge vega (TODO) ---
        # Calculate target quantities and execute at bid/ask (no upfront premium).

        # --- 5. Mark existing futures and rebalance the delta hedge ---
        # TODO: Include the delta of actual option holdings in each futures target.
        futures_pnl = 0.0
        execution_pnl = 0.0
        for contract, delta in zip(contracts, (delta_april, delta_may)):
            old_position = positions[contract]
            target = target_futures(delta, barrels, lot_size)
            if old_position == 0 and target == 0:
                continue

            # Yesterday's position earns today's price movement before rebalancing.
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

        # --- 6. Update cash, portfolio value and daily P&L ---
        # TODO: Include futures-style option daily settlements in cash, without double counting.
        cash += futures_pnl + execution_pnl + interest_pnl
        rows.append({
            **option_results,
            "date": today,
            "rho": rho,
            "discount_factor": daily_arguments["DF"],
            "interest_pnl": interest_pnl,
            "collar_value_per_barrel": value,
            "delta_april": delta_april,
            "delta_may": delta_may,
            "vega_april_per_barrel": vega_april,
            "vega_may_per_barrel": vega_may,
            "april_vol_up_1bp_pnl_usd": barrels * vega_april * 0.0001,
            "may_vol_up_1bp_pnl_usd": barrels * vega_may * 0.0001,
            "position_april": positions[contracts[0]],
            "position_may": positions[contracts[1]],
            "collar_pnl": collar_pnl,
            "futures_pnl": futures_pnl,
            "execution_pnl": execution_pnl,
            "delta_hedged_pnl": collar_pnl + futures_pnl + execution_pnl + interest_pnl,
            "cash": cash,
            "portfolio_value": barrels * value + cash,
        })
        previous_value = value

    # --- Assemble results and reconcile P&L with changes in portfolio value ---
    result = pd.DataFrame(rows).set_index("date")
    # Monitora la contabilità anche sui dati reali; verificato nei test contabili.
    result["accounting_error"] = check_accounting(result)
    return result
