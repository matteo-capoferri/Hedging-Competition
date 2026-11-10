import pandas as pd


def build_fixing_schedule(fixing_dates, front_code, next_code, front_expiry):
    """Resolve the Annex B roll using supplied exchange sessions and expiry."""
    dates = pd.DatetimeIndex(fixing_dates).sort_values()
    if dates.hasnans or dates.has_duplicates:
        raise ValueError("Le date di fixing devono essere valide e univoche")
    prior = dates[dates < pd.Timestamp(front_expiry)]
    if len(prior) < 2:
        raise ValueError("Servono almeno due fixing prima della scadenza del front")
    last_front = prior[-2]
    return pd.DataFrame({
        "date": dates,
        "future_contract_code": [front_code if date <= last_front else next_code for date in dates],
        "weight": 1.0 / len(dates),
    })


def training_fixing_schedule():
    # Same provisional weekday calendar as the original training prototype.
    return build_fixing_schedule(
        pd.bdate_range("2021-02-01", "2021-02-26"), "COJ21", "COK21", "2021-02-26",
    )


def build_calendar(emission_date):
    """
    INPUT:
        emission_date = starting date of the simulation.

    OUTPUT:
        fixing = starting date followed by the April and May fixing dates.
        time_steps = time between consecutive dates, in years using days / 365.

    Build the fixing calendar using weekdays; exchange holidays are not excluded.
    """
    fixing = pd.DatetimeIndex(training_fixing_schedule().date).insert(0, emission_date)
    time_steps = fixing.to_series().diff().dropna().dt.days.to_numpy() / 365
    return fixing, time_steps
