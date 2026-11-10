import numpy as np
from scipy.optimize import brentq

from black76 import black_76


def extract_smile(options):
    """Calcola, per data, gli strike ordinati e le IV medie delle opzioni filtrate."""
    return {
        day: _smile_for_snapshot(options.loc[options["date"] == day])
        for day in options["date"].unique()
    }


def _implied_volatility(row):
    future, strike = row.underlying_settlement, row.strike
    if row.option_type == "Call":
        intrinsic_value = max(future - strike, 0)
    else:
        intrinsic_value = max(strike - future, 0)

    if abs(row.settlement_price - intrinsic_value) < 1e-10:
        return 0.0

    return brentq(
        lambda sigma: black_76(
            future, strike, row.time_to_expiry_days / 365, sigma, row.option_type,
        ) - row.settlement_price,
        1e-10, 5,
    )


def _smile_for_snapshot(options):
    volatilities = [_implied_volatility(row) for row in options.itertuples(index=False)]
    smile = options[["strike"]].assign(IV=volatilities)
    mean_volatilities = smile.groupby("strike")["IV"].mean()
    return {"K": mean_volatilities.index.tolist(), "IV": mean_volatilities.tolist()}


def build_smiles(quotes, emission_date, contracts):
    """Estrae uno smile alla data richiesta per ciascun contratto."""
    return _smiles_by_contract(quotes.loc[quotes["date"] == emission_date], contracts)


def _smiles_by_contract(quotes, contracts):
    smiles = {}
    for contract in contracts:
        options = quotes.loc[quotes["actual_underlying_ticker"] == contract]
        if options.empty:
            raise ValueError(f"Nessuna opzione idonea per {contract}")
        smiles[contract] = _smile_for_snapshot(options)
    return smiles


def initial_conditions(smiles, spot, contracts):
    """Restituisce prezzi e volatilità iniziali nello stesso ordine dei contratti."""
    settlements = spot[["future_ticker", "settlement"]].dropna().drop_duplicates()
    prices = settlements.set_index("future_ticker", verify_integrity=True)["settlement"]
    prices = prices.reindex(contracts)
    if not prices.between(0, float("inf"), inclusive="neither").fillna(False).all():
        raise ValueError("Servono settlement iniziali positivi e finiti per entrambi i futures")

    volatilities = tuple(
        np.interp(prices[contract], smiles[contract]["K"], smiles[contract]["IV"])
        for contract in contracts
    )
    return tuple(prices), volatilities


def prepare_smiles(quotes, spot, contracts):
    """Calcola smile, prezzi e volatilità iniziali dai dati già filtrati per data."""
    smiles = _smiles_by_contract(quotes, contracts)
    emission_date = quotes["date"].iloc[0]
    initial_spot = spot.loc[spot["date"] == emission_date]
    prices, volatilities = initial_conditions(smiles, initial_spot, contracts)
    return smiles, prices, volatilities
