import pandas as pd
from scipy.optimize import brentq

from black76 import black_76


def extract_smile(options):
    """
    INPUT:
        options = option quotes for one futures contract on one market date.
            strike = option strike price.
            underlying_settlement = futures settlement price.
            time_to_expiry_days = number of calendar days until expiry.
            option_type = "Call" or "Put".
            settlement_price = observed option price.

    OUTPUT:
        K = list of unique strikes in increasing order.
        IV = list of mean implied volatilities, in the same order as K.
        Both lists are returned in one dictionary with keys "K" and "IV".
        Volatilities are annualized decimals: 0.20 means 20%.

    Calculate Black-76 implied volatilities and average values at the same strike.
    Quotes with missing option prices are excluded.
    """
    options = options.dropna(subset=["settlement_price"]).sort_values("strike")
    strikes = []
    implied_volatilities = []
    for row in options.itertuples(index=False):
        volatility = brentq(
            lambda sigma: black_76(
                row.underlying_settlement, row.strike,
                row.time_to_expiry_days / 365, sigma, row.option_type,
            ) - row.settlement_price,
            1e-5, 3,
        )
        strikes.append(row.strike)
        implied_volatilities.append(volatility)
    smile = pd.DataFrame({"K": strikes, "IV": implied_volatilities})
    smile = smile.groupby("K", as_index=False)["IV"].mean().sort_values("K")
    return {"K": smile["K"].tolist(), "IV": smile["IV"].tolist()}
