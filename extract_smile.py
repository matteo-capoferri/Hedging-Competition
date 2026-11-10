import pandas as pd
from scipy.optimize import brentq

from black76 import black_76


def extract_smile(quotes):
    """
    INPUT:
        quotes = filtered option quotes for all contracts and market dates.

    OUTPUT:
        smiles = dictionary organized as smiles[contract][day].
            K = unique strikes in increasing order.
            IV = mean implied volatilities at those strikes, as decimals.

    Calculate one smile per contract and day.
    Average call and put IV values when they share the same strike.
    Input prices must exceed intrinsic value, as enforced by preprocessing.
    """
    smiles = {}

    for contract in quotes["actual_underlying_ticker"].unique():
        contract_quotes = quotes.loc[quotes["actual_underlying_ticker"] == contract]
        smiles[contract] = {}

        for day in sorted(contract_quotes["date"].unique()):
            daily_quotes = contract_quotes.loc[contract_quotes["date"] == day]
            strikes = []
            volatilities = []

            for row in daily_quotes.itertuples(index=False):
                volatility = brentq(
                    lambda sigma: black_76(
                        row.underlying_settlement,
                        row.strike,
                        row.time_to_expiry_days / 365,
                        sigma,
                        row.option_type,
                    ) - row.settlement_price,
                    1e-10, 5,
                )

                strikes.append(row.strike)
                volatilities.append(volatility)

            daily_smile = pd.DataFrame({"K": strikes, "IV": volatilities})
            daily_smile = daily_smile.groupby("K", as_index=False)["IV"].mean()
            daily_smile = daily_smile.sort_values("K")

            smiles[contract][day] = {
                "K": daily_smile["K"].tolist(),
                "IV": daily_smile["IV"].tolist(),
            }

    return smiles
