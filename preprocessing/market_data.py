import pandas as pd

from get_initial_price import get_initial_price
from preprocessing.options import filter_option_quotes, filter_smile_quotes


def prepare_market_data(data_dir, emission_date, contracts):
    """Restituisce le opzioni filtrate e i settlement futures alla data di emissione."""
    april, may = contracts
    quotes = pd.read_parquet(
        data_dir / "unpivoted_bid_offer_train_clean.parquet",
        columns=["date", "actual_underlying_ticker", "strike", "option_type",
                 "settlement_price", "underlying_settlement", "time_to_expiry_days",
                 "option_bid", "option_ask"],
    )
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes = quotes.loc[
        (quotes["date"] == emission_date) & quotes["actual_underlying_ticker"].isin(contracts)
    ]

    april_history = pd.read_excel(
        data_dir / "COJ21_COZ22_historical_series.xlsx", sheet_name=april,
    )
    april_dates = pd.to_datetime(april_history["date"])
    april_price = april_history.loc[april_dates == emission_date, "settlement"].item()
    may_price = get_initial_price(quotes.loc[quotes["actual_underlying_ticker"] == may])
    spot = pd.DataFrame({
        "date": emission_date,
        "actual_underlying_ticker": contracts,
        "settlement": [april_price, may_price],
    })
    if not spot["settlement"].between(0, float("inf"), inclusive="neither").all():
        raise ValueError("I settlement iniziali dei futures devono essere positivi e finiti")

    return filter_smile_quotes(filter_option_quotes(quotes)), spot
