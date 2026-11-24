import warnings
import numpy as np


def select_option(today, contract, quotes, previous=None):
    """
    INPUT:
        today = current market date.
        contract = underlying future to hedge.
        quotes = filtered option quotes with ticker and expiration.
        previous = previously selected row, or None.
    OUTPUT:
        option = selected row, or None if no suitable option exists.
            date remains the original quote date when carrying prices forward.

    Keep the option only while at least 15 days remain and its strike is nearest
    to today's future price among available strikes on the same future and expiry.
    Equal-distance strikes are both acceptable, avoiding unnecessary switches.
    New selections require 30 days; rank by ATM distance, then bid-ask spread.
    Try the requested future first, then positive correlations in descending order.
    Correlations use up to 60 recent sessions and at least 20 common log returns.
    """
    daily = quotes.loc[quotes["date"] == today]
    if previous is not None and (previous["expiration"] - today).days >= 15:
        current = daily.loc[daily["option_ticker"] == previous["option_ticker"]]
        alternatives = daily.loc[
            (daily["actual_underlying_ticker"] == previous["actual_underlying_ticker"])
            & (daily["expiration"] == previous["expiration"])
        ]
        still_atm = True
        if not alternatives.empty:
            future_price = alternatives["underlying_settlement"].iloc[0]
            nearest_distance = abs(alternatives["strike"] - future_price).min()
            current_distance = abs(previous["strike"] - future_price)
            still_atm = current_distance <= nearest_distance + 1e-10
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
