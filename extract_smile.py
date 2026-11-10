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
    
    smiles = {}
    for day in options["date"].unique():
        for row in options.loc[options["date"]==day].itertuples(index=False):
            #check if vol is zero
            if row.option_type == "Call":
                intrinsic_value = max(row.underlying_settlement - row.strike, 0)
            else:
                intrinsic_value = max(row.strike - row.underlying_settlement, 0)

            if abs(row.settlement_price - intrinsic_value) < 1e-10:
                volatility = 0.0
            else:
                difference_low = black_76(
                    row.underlying_settlement, row.strike,
                    row.time_to_expiry_days / 365, 1e-5, row.option_type,
                ) - row.settlement_price

                difference_high = black_76(
                    row.underlying_settlement, row.strike,
                    row.time_to_expiry_days / 365, 5, row.option_type,
                ) - row.settlement_price

                if difference_low * difference_high > 0:
                    print(day, row)
                    print("Price differences:", difference_low, difference_high)
                volatility = brentq(
                    lambda sigma: black_76(
                        row.underlying_settlement, row.strike,
                        row.time_to_expiry_days / 365, sigma, row.option_type,
                    ) - row.settlement_price,
                    1e-10, 5,
                )
        strikes = []
        implied_volatilities = []
        strikes.append(row.strike)
        implied_volatilities.append(volatility)
        smile = pd.DataFrame({"K": strikes, "IV": implied_volatilities})
        smile = smile.groupby("K", as_index=False)["IV"].mean().sort_values("K")
        smiles[day] = {"K": smile["K"].tolist(), "IV": smile["IV"].tolist()}
        
    return smiles
