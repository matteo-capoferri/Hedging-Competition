"""Interfaccia storica per estrarre lo smile dai settlement delle opzioni."""

from black76 import black76, implied_volatility


def extract_smile(options):
    """Raggruppa per data e conserva tutti gli strike con settlement."""
    smiles = {}
    quotes = options.dropna(subset=["settlement_price"]).copy()
    for date, daily in quotes.groupby("date", sort=False):
        points = daily.assign(iv=daily.apply(_settlement_iv, axis=1))
        by_strike = points.groupby("strike").iv.mean()
        smiles[date] = {"K": by_strike.index.tolist(), "IV": by_strike.tolist()}
    return smiles


def _settlement_iv(row):
    maturity = row.time_to_expiry_days / 365.0
    args = (row.underlying_settlement, row.strike, maturity)
    intrinsic = black76(*args, 0.0, row.option_type)
    if abs(row.settlement_price - intrinsic) < 1e-10:
        return 0.0
    return implied_volatility(
        *args, row.option_type, row.settlement_price, lower=1e-10, upper=5.0,
    )
