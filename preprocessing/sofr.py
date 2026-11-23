import pandas as pd


def prepare_sofr(csv_path, dates):
    """Allinea fixing pubblicati e crescita della cassa alle sedute del backtest."""
    dates = pd.DatetimeIndex(dates)
    data = pd.read_csv(csv_path, parse_dates=["observation_date"], na_values=".")
    fixings = data.set_index("observation_date")["SOFR"].dropna().sort_index() / 100
    if fixings.empty or fixings.index[0] >= dates[0] or fixings.index[-1] < dates[-1]:
        raise ValueError("Servono i fixing SOFR dal giorno precedente fino a fine backtest.")

    # Un fixing diventa pubblico il giorno lavorativo SOFR successivo.
    published = fixings.shift(1)
    known_rate = published.reindex(dates, method="ffill")
    if known_rate.isna().any():
        raise ValueError("Manca un fixing SOFR già pubblicato alla prima valutazione.")

    # Indice realizzato ACT/360: il fixing del venerdì copre anche il weekend.
    days = fixings.index.to_series().diff().dt.days
    growth = 1 + fixings.shift(1) * days / 360
    growth.iloc[0] = 1.0
    cash_index = growth.cumprod()
    # Interesse semplice tra due fixing, anche se nel mezzo il Brent è aperto.
    calendar = cash_index.index.union(dates)
    cash_index = cash_index.reindex(calendar).interpolate(method="time").loc[dates]
    cash_growth = cash_index / cash_index.shift(1)
    cash_growth.iloc[0] = 1.0
    return pd.DataFrame({"known_rate": known_rate, "cash_growth": cash_growth})


def discount_factor(today, payment_date, annual_rate):
    """Curva piatta al tasso noto, con capitalizzazione giornaliera ACT/360."""
    remaining_days = (payment_date - today).days
    return (1 + annual_rate / 360) ** (-remaining_days)
