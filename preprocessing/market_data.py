import pandas as pd

from preprocessing.options import filter_option_quotes, filter_smile_quotes


def prepare_market_data(data_dir, emission_date, contracts):
    """Return filtered option quotes and futures history indexed by date and ticker."""
    quotes = pd.read_parquet(
        data_dir / "unpivoted_bid_offer_train_clean.parquet",
        columns=["date", "actual_underlying_ticker", "strike", "option_type",
                 "settlement_price", "underlying_settlement", "time_to_expiry_days",
                 "option_bid", "option_ask", "underlying_bid", "underlying_ask"],
    )
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes = quotes.loc[quotes["actual_underlying_ticker"].isin(contracts)]
    spot = load_futures(data_dir, quotes, contracts)
    spot = settlement_for(spot)
    spot = spot.set_index(["date", "future_ticker"])

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


def settlement_for(spot):
    """
    INPUT:
        spot = futures quotes, possibly with several rows for the same day.

    OUTPUT:
        clean_spot = one row per date and ticker, with one settlement price.
        Bid and ask are kept only if their available values agree.
    """
    rows = []
    for (date, ticker), quotes in spot.groupby(["date", "future_ticker"]):
        settlements = quotes["settlement"].dropna().unique()
        if len(settlements) != 1:
            raise ValueError(f"Missing or conflicting settlement for {ticker} on {date}")

        bids = quotes["bid"].dropna().unique()
        asks = quotes["ask"].dropna().unique()
        bid = bids[0] if len(bids) == 1 else float("nan")
        ask = asks[0] if len(asks) == 1 else float("nan")
        rows.append([date, ticker, settlements[0], bid, ask])

    clean_spot = pd.DataFrame(rows, columns=["date", "future_ticker", "settlement", "bid", "ask"])
    return clean_spot
