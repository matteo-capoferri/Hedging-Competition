import pandas as pd


def build_calendar(emission_date):
    """Restituisce i fixing feriali e gli intervalli in anni dalla data di emissione."""
    fixing_apr21 = pd.bdate_range("2021-02-01", "2021-02-24")
    fixing_may21 = pd.bdate_range("2021-02-25", "2021-02-26")
    fixing = fixing_apr21.append(fixing_may21).insert(0, emission_date)
    time_steps = fixing.to_series().diff().dropna().dt.days.to_numpy() / 365
    return fixing, time_steps
