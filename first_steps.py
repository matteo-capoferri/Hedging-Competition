"""Carica i dati di training, controlla le quotazioni e calcola le IV Black-76.

Esecuzione dalla cartella del progetto: python first_steps.py
I file originali non vengono modificati.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from black76 import black76, implied_vol
from market_data import build_future_quotes, default_data_dir, load_data
from option_analytics import Config, daily_iv_panel, prepare_quotes


def future_spread_counts(quotes: pd.DataFrame) -> pd.DataFrame:
    columns = ["underlying_bid", "underlying_ask"]
    return quotes.groupby(["date", "future_contract_code"])[columns].nunique()


def futures_panel(
    quotes: pd.DataFrame,
    history: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    counts = future_spread_counts(quotes)
    if counts[["underlying_bid", "underlying_ask"]].gt(1).any().any():
        warnings.warn(
            "Bid/ask futures discordanti tra righe. "
            "I valori non vengono aggregati automaticamente."
        )
    return build_future_quotes(quotes, history).settlement.unstack("future_contract_code")


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
    spread_counts = future_spread_counts(raw)
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
            "Questo report copre i dati e le IV; main.py esegue il pricing iniziale del collar.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Risultati salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "initial_analysis")
    parser.add_argument("--date", default=Config.valuation_date)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, Config(valuation_date=args.date))
