"""Filtri di liquidità, volatilità implicite e punti dello smile."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from black76 import implied_vol


@dataclass(frozen=True)
class Config:
    valuation_date: str = "2020-01-06"
    training_end: str = "2021-04-06"
    legs: tuple[str, str] = ("COJ21", "COK21")
    # Soglie esplorative, da validare prima di congelare la strategia.
    max_relative_spread: float = 0.50
    min_mid: float = 0.05
    correlation_window: int = 60
    correlation_min_periods: int = 20


def prepare_quotes(quotes: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Conserva tutte le righe: i motivi di esclusione restano ispezionabili."""
    q = quotes.copy()
    q["ttm"] = (q.expiration - q.date).dt.days / 365.0
    q["option_mid"] = (q.option_bid + q.option_ask) / 2
    q["option_spread"] = q.option_ask - q.option_bid
    q["relative_spread"] = q.option_spread / q.option_mid.where(q.option_mid > 0)
    required_prices = ["strike", "underlying_settlement", "option_bid", "option_ask"]
    finite_prices = np.isfinite(q[required_prices]).all(axis=1)
    q["valid_market"] = (
        finite_prices
        & q.option_type.isin(["Call", "Put"])
        & (q.strike > 0)
        & (q.underlying_settlement > 0)
        & (q.ttm > 0)
        & (q.option_bid >= 0)
        & (q.option_ask >= q.option_bid)
    )
    option_sign = np.where(q.option_type == "Call", 1, -1)
    intrinsic = np.maximum(
        option_sign * (q.underlying_settlement - q.strike), 0
    )
    upper = np.where(q.option_type == "Call", q.underlying_settlement, q.strike)
    q["valid_bounds"] = (q.option_mid > intrinsic + 1e-10) & (q.option_mid < upper)
    q["liquid"] = (
        (q.option_mid >= config.min_mid)
        & (q.relative_spread <= config.max_relative_spread)
    )
    q["is_otm"] = np.where(
        q.option_type == "Call",
        q.strike >= q.underlying_settlement,
        q.strike < q.underlying_settlement,
    )
    q["eligible"] = q.valid_market & q.valid_bounds & q.liquid
    return q


def daily_iv_panel(quotes: pd.DataFrame) -> pd.DataFrame:
    """Compute IV and smile coordinates using only each row's market snapshot."""
    q = quotes.copy()
    q["iv_mid"] = np.nan
    q["log_moneyness"] = np.nan
    mask = q.eligible & q.is_otm
    if mask.any():
        selected = q.loc[mask]
        q.loc[mask, "log_moneyness"] = np.log(selected.strike / selected.underlying_settlement)
        q.loc[mask, "iv_mid"] = selected.apply(implied_vol, axis=1)
    q["total_variance"] = q.ttm * q.iv_mid.pow(2)
    return q


def atm_smile(ivs: pd.DataFrame, future_code: str, forward: float) -> tuple[pd.Series, float]:
    """Interpola la IV al settlement usando i punti di una sola scadenza."""
    smile = ivs.loc[ivs.future_contract_code.eq(future_code) & ivs.iv_mid.notna()]
    points = smile.groupby("strike").iv_mid.mean().sort_index()
    if len(points) < 2 or smile.expiration.nunique() != 1:
        raise ValueError(f"Input smile insufficienti o su scadenze diverse per {future_code}")
    if not np.isfinite(forward) or not points.index.min() <= forward <= points.index.max():
        raise ValueError(f"Settlement fuori dall'intervallo degli strike per {future_code}")
    return points, float(np.interp(forward, points.index, points))
