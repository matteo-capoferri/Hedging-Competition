"""Carica i dati di training, controlla le quotazioni e calcola le IV Black-76.

Esecuzione dalla cartella del progetto: python first_steps.py
I file originali non vengono modificati.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr


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


def black76(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    option_type: str,
) -> float:
    """Prezzo Black-76 non scontato, con tempo ACT/365F."""
    if option_type not in {"Call", "Put"}:
        raise ValueError(f"Tipo opzione non riconosciuto: {option_type}")
    if not np.isfinite([forward, strike, maturity, sigma]).all():
        raise ValueError("Gli input Black devono essere finiti")
    if forward <= 0 or strike <= 0 or maturity < 0 or sigma < 0:
        raise ValueError("Black richiede F,K positivi e T,sigma non negativi")
    option_sign = 1 if option_type == "Call" else -1
    if maturity == 0 or sigma == 0:
        return max(option_sign * (forward - strike), 0.0)
    v = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strike) / v + v / 2
    d2 = d1 - v
    return float(
        option_sign
        * (forward * ndtr(option_sign * d1) - strike * ndtr(option_sign * d2))
    )


def implied_vol(row: pd.Series) -> float:
    """NaN se il prezzo non ammette una radice nel bracket numerico."""
    forward = row.underlying_settlement
    strike = row.strike
    maturity = row.ttm
    option_type = row.option_type
    market_price = row.option_mid
    try:
        return brentq(
            lambda volatility: black76(
                forward, strike, maturity, volatility, option_type
            )
            - market_price,
            1e-8,
            10.0,
            xtol=1e-10,
        )
    except ValueError:
        return np.nan


def load_data(data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    quotes = pd.read_parquet(data_dir / "unpivoted_bid_offer_train_clean.parquet")
    required = {
        "date", "expiration", "future_contract_code", "option_ticker", "strike",
        "option_type", "option_bid", "option_ask", "underlying_settlement",
        "underlying_bid", "underlying_ask", "sofr_rate",
    }
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")
    for column in ("date", "expiration"):
        quotes[column] = pd.to_datetime(quotes[column], errors="raise")
        if quotes[column].isna().any():
            raise ValueError(f"Date mancanti nella colonna {column}")
    if quotes[["option_ticker", "future_contract_code"]].isna().any().any():
        raise ValueError("Ticker o codici contratto mancanti")
    quote_key = ["date", "option_ticker"]
    if quotes.duplicated(quote_key).any():
        raise ValueError("Quotazioni duplicate per data e ticker: verificare la fonte")
    history = pd.read_excel(
        data_dir / "COJ21_COZ22_historical_series.xlsx", sheet_name=None,
    )
    for name, frame in history.items():
        if not {"date", "settlement", "bid", "ask"}.issubset(frame.columns):
            raise ValueError(f"Schema inatteso nel foglio {name}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame.date.isna().any():
            raise ValueError(f"Date mancanti nel foglio {name}")
        if frame.date.duplicated().any():
            raise ValueError(f"Date duplicate nel foglio {name}")
        history[name] = frame.sort_values("date").reset_index(drop=True)
    quotes = quotes.sort_values(quote_key).reset_index(drop=True)
    return quotes, history


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
    # Il SOFR grezzo sembra espresso in percentuale (es. 1.55).
    # Non costruire una curva a termine da questa sola colonna.
    return q


def futures_panel(
    quotes: pd.DataFrame,
    history: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = ["underlying_settlement", "underlying_bid", "underlying_ask"]
    grouped = quotes.groupby(["date", "future_contract_code"])
    counts = grouped[columns].nunique()
    if counts.underlying_settlement.gt(1).any():
        raise ValueError("Settlement futures incoerenti nella stessa data/contratto")
    if counts[["underlying_bid", "underlying_ask"]].gt(1).any().any():
        warnings.warn(
            "Bid/ask futures discordanti tra righe. "
            "I valori non vengono aggregati automaticamente."
        )
    panel = grouped.underlying_settlement.first().unstack("future_contract_code")
    for contract, frame in history.items():
        series = frame.set_index("date").settlement
        if contract in panel:
            overlap = pd.concat([panel[contract], series], axis=1, sort=True).dropna()
            if not np.allclose(overlap.iloc[:, 0], overlap.iloc[:, 1], atol=1e-8, rtol=0):
                raise ValueError(f"Excel e Parquet non concordano per {contract}")
        panel = panel.reindex(panel.index.union(series.index))
        panel[contract] = panel.get(contract, pd.Series(dtype=float)).combine_first(series)
    return panel.sort_index()


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


def smile_coverage(quotes: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Track smile inputs per day and leg, including futures-only dates."""
    keys = ["date", "future_contract_code"]
    quote_counts = quotes.groupby(keys).size().rename("quote_rows")
    eligible_counts = quotes.loc[quotes.eligible & quotes.is_otm].groupby(keys).size().rename("eligible_otm")
    solved = quotes.loc[quotes.iv_mid.notna()].groupby(keys).agg(
        iv_count=("iv_mid", "size"), distinct_strikes=("strike", "nunique"),
        expirations=("expiration", "nunique"), k_min=("log_moneyness", "min"),
        k_max=("log_moneyness", "max"),
    )
    active = prices.rename_axis(index="date", columns="future_contract_code").notna()
    active_index = active.stack().loc[lambda is_active: is_active].index
    coverage = pd.concat([quote_counts, eligible_counts, solved], axis=1)
    coverage = coverage.reindex(active_index.union(coverage.index)).sort_index()
    count_columns = ["quote_rows", "eligible_otm", "iv_count", "distinct_strikes", "expirations"]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    return coverage


def run(data_dir: Path, output_dir: Path, config: Config) -> None:
    valuation_date = pd.Timestamp(config.valuation_date)
    training_end = pd.Timestamp(config.training_end)
    if pd.isna(valuation_date) or pd.isna(training_end) or valuation_date > training_end:
        raise ValueError("Data di valutazione non valida o successiva al training")
    if not 1 < config.correlation_min_periods <= config.correlation_window:
        raise ValueError("Richiesto 1 < correlation_min_periods <= correlation_window")
    raw, history = load_data(data_dir)
    # Nessun utilizzo di osservazioni oltre la fine del training.
    raw = raw.loc[raw.date <= pd.Timestamp(config.training_end)].copy()
    history = {k: v.loc[v.date <= pd.Timestamp(config.training_end)] for k, v in history.items()}
    q = prepare_quotes(raw, config)
    panel = futures_panel(raw, history)
    for leg in config.legs:
        if leg not in panel:
            raise ValueError(f"Gamba assente nei dati: {leg}")
    # Niente forward-fill: non inventare osservazioni o rendimenti.
    prices = panel[list(config.legs)].where(lambda x: x > 0)
    returns = np.log(prices / prices.shift(1))
    rho = returns.iloc[:, 0].rolling(
        config.correlation_window, min_periods=config.correlation_min_periods,
    ).corr(returns.iloc[:, 1]).rename("rho")
    leg_quotes = q.loc[q.future_contract_code.isin(config.legs)]
    today = leg_quotes.loc[leg_quotes.date.eq(valuation_date)]
    if today.empty:
        raise ValueError("Nessuna quotazione per le gambe e la data richieste")
    missing_legs = set(config.legs) - set(today.future_contract_code)
    if missing_legs:
        raise ValueError(f"Quotazioni mancanti alla data scelta: {sorted(missing_legs)}")
    daily = daily_iv_panel(leg_quotes)
    today = daily.loc[daily.date.eq(valuation_date)]
    mask = today.eligible & today.is_otm
    smile_inputs = daily.loc[daily.iv_mid.notna()]
    smile_summary = smile_coverage(daily, prices.loc[raw.date.min():])
    output_dir.mkdir(parents=True, exist_ok=True)
    today.to_csv(output_dir / "initial_quotes_and_iv.csv", index=False)
    smile_inputs.to_csv(output_dir / "daily_smile_inputs.csv", index=False)
    smile_summary.to_csv(output_dir / "daily_smile_coverage.csv")
    panel.to_csv(output_dir / "futures_settlements.csv")
    pd.concat([returns.add_prefix("log_return_"), rho], axis=1).to_csv(output_dir / "returns_and_correlation.csv")
    coverage = q.groupby("future_contract_code").agg(
        first_date=("date", "min"), last_date=("date", "max"),
        rows=("date", "size"), eligible=("eligible", "sum"),
    )
    coverage.to_csv(output_dir / "data_coverage.csv")
    spread_counts = raw.groupby(["date", "future_contract_code"])[["underlying_bid", "underlying_ask"]].nunique()
    conflicts = spread_counts.loc[spread_counts.gt(1).any(axis=1)]
    conflicts.to_csv(output_dir / "futures_quote_conflicts.csv")
    report = {
        "rows": len(q), "valuation_date": config.valuation_date,
        "eligible_rows": int(q.eligible.sum()),
        "initial_otm_iv_count": int(today.iv_mid.notna().sum()),
        "initial_iv_failures": int((mask & today.iv_mid.isna()).sum()),
        "daily_otm_iv_count": len(smile_inputs),
        "daily_iv_failures": int((daily.eligible & daily.is_otm & daily.iv_mid.isna()).sum()),
        "futures_bid_ask_conflict_groups": len(conflicts),
        "initial_paired_returns": int(returns.loc[:config.valuation_date].dropna().shape[0]),
        "notes": [
            "La correlazione rimane NaN finche' non ci sono abbastanza osservazioni passate.",
            "Nessuna sostituzione di bid/ask mancanti con settlement.",
            "IV: Black-76 non scontata; premio di esercizio anticipato assunto nullo nel modello ideale futures-style.",
            "Non sono ancora implementati smile, calendario ICE, collar, hedge o curva SOFR.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Risultati salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "initial_analysis")
    parser.add_argument("--date", default=Config.valuation_date)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, Config(valuation_date=args.date))
