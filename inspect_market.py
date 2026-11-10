"""Mostra quotazioni e fixing, con esportazione CSV facoltativa."""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from build_calendar import training_fixing_schedule
from market_data import MarketData, default_data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--date", type=date.fromisoformat, help="Filtra le quotazioni (AAAA-MM-GG)")
    parser.add_argument("--contract", type=str.upper, help="Ticker del future, es. COJ21")
    parser.add_argument("--as-of", type=date.fromisoformat,
                        help="Data di osservazione dei fixing (AAAA-MM-GG)")
    parser.add_argument("--output-dir", type=Path, help="Cartella per esportare i CSV")
    args = parser.parse_args()

    market = MarketData.load(args.data_dir)
    futures = market.future_quotes.reset_index()
    options = market.option_quotes.reset_index()
    sofr = market.sofr.rename("sofr_rate_raw").reset_index()
    if args.date:
        session = pd.Timestamp(args.date)
        futures = futures.loc[futures.date.eq(session)]
        options = options.loc[options.date.eq(session)]
        sofr = sofr.loc[sofr.date.eq(session)]
    if args.contract:
        futures = futures.loc[futures.future_contract_code.eq(args.contract)]
        options = options.loc[options.future_contract_code.eq(args.contract)]

    schedule = training_fixing_schedule()
    as_of = pd.Timestamp(args.as_of or args.date or schedule.date.max())
    fixings = market.fixing_table(schedule, as_of)
    print(f"Fixing alla chiusura del {as_of.date()} (calendario provvisorio)")
    print(fixings.to_string(index=False, na_rep="-"))
    print(f"Contributo dei fixing osservati: {fixings.contribution.sum():.4f} USD/bbl")

    option_columns = [
        "date", "option_ticker", "future_contract_code", "expiration", "strike",
        "option_type", "option_bid", "option_ask", "settlement_price",
    ]
    # Mantieni tutte le colonne originali, con i campi principali in testa.
    option_columns = [column for column in option_columns if column in options]
    extra_columns = [column for column in options if column not in option_columns]
    options = options[option_columns + extra_columns]
    tables = {"futures": futures, "opzioni": options, "sofr": sofr, "fixing": fixings}
    for name, table in tables.items():
        print(f"{name}: {len(table)} righe")
        if args.date and name != "fixing" and not table.empty:
            preview = table[option_columns] if name == "opzioni" else table
            print(preview.head(8).to_string(index=False, na_rep="-"))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"CSV salvati in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
