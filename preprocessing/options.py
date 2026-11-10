import pandas as pd


def filter_option_quotes(quotes, min_mid=0.05, max_relative_spread=0.50):
    """Seleziona le opzioni in vita con un mercato sufficientemente liquido."""
    days = quotes["time_to_expiry_days"]
    bid = quotes["option_bid"]
    ask = quotes["option_ask"]
    valid = (days > 0) & (days < float("inf"))
    valid &= (bid >= 0) & (ask >= bid) & (ask < float("inf"))
    quotes = quotes.loc[valid]

    bid = quotes["option_bid"]
    ask = quotes["option_ask"]
    mid = (bid + ask) / 2
    spread = ask - bid
    enough_premium = (mid > 0) & (mid >= min_mid)
    return quotes.loc[enough_premium & (spread <= max_relative_spread * mid)].copy()


def filter_smile_quotes(quotes):
    """Seleziona strike, future e settlement validi per la volatilità implicita."""
    strike = quotes["strike"]
    future = quotes["underlying_settlement"]
    price = quotes["settlement_price"]

    valid_inputs = (strike > 0) & (strike < float("inf"))
    valid_inputs &= (future > 0) & (future < float("inf"))

    intrinsic = pd.Series(0.0, index=quotes.index)
    upper = pd.Series(0.0, index=quotes.index)

    calls = quotes["option_type"] == "Call"
    puts = quotes["option_type"] == "Put"

    intrinsic.loc[calls] = future.loc[calls] - strike.loc[calls]
    upper.loc[calls] = future.loc[calls]

    intrinsic.loc[puts] = strike.loc[puts] - future.loc[puts]
    upper.loc[puts] = strike.loc[puts]

    intrinsic.loc[intrinsic < 0] = 0

    quotes = quotes.loc[
        valid_inputs
        & (calls | puts)
        & (price > intrinsic + 1e-10)
        & (price < upper)
    ]
    return quotes
