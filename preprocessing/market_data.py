import pandas as pd

from preprocessing.options import filter_option_quotes, filter_smile_quotes


def prepare_market_data(data_dir, emission_date, contracts):
    """Restituisce le opzioni all'emissione e le serie storiche dei futures."""
    quotes = pd.read_parquet(
        data_dir / "unpivoted_bid_offer_train_clean.parquet",
        columns=["date", "actual_underlying_ticker", "strike", "option_type",
                 "settlement_price", "underlying_settlement", "time_to_expiry_days",
                 "option_bid", "option_ask", "underlying_bid", "underlying_ask"],
    )
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes = quotes.loc[quotes["actual_underlying_ticker"].isin(contracts)]
    spot = load_futures(data_dir, quotes, contracts)


    return filter_smile_quotes(filter_option_quotes(quotes)), spot


def load_futures(data_dir, quotes, contracts):
    """Unisce le osservazioni dei futures da Excel e Parquet."""
    april, may = contracts
    columns = ["date", "future_ticker", "settlement", "bid", "ask"]

    april_spot = pd.read_excel(
        data_dir / "COJ21_COZ22_historical_series.xlsx",
        sheet_name=april, usecols=["date", "settlement", "bid", "ask"],
        parse_dates=["date"],
    ).assign(future_ticker=april)

    may_spot = quotes.loc[quotes["actual_underlying_ticker"] == may].rename(
        columns={
            "actual_underlying_ticker": "future_ticker",
            "underlying_settlement": "settlement",
            "underlying_bid": "bid",
            "underlying_ask": "ask",
        }
    )

    spot = pd.concat([april_spot[columns], may_spot[columns]], ignore_index=True)
    return spot.drop_duplicates().sort_values(["date", "future_ticker"], ignore_index=True)
