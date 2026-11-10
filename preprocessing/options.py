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
    kind = quotes["option_type"]
    price = quotes["settlement_price"]

    valid_inputs = True
    # valid_inputs = kind.isin(["Call", "Put"])
    valid_inputs &= (strike > 0) & (strike < float("inf"))
    valid_inputs &= (future > 0) & (future < float("inf"))

    is_call = (kind == "Call").fillna(False)
    intrinsic = (future - strike).where(is_call, strike - future).clip(lower=0)
    upper = future.where(is_call, strike)
    return quotes.loc[valid_inputs & (price >= intrinsic) & (price < upper)]
