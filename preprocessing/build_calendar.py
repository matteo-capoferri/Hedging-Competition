import pandas as pd


def build_calendar(valuation_date, contracts):
    april, may = contracts
    april_dates = pd.bdate_range("2021-02-01", "2021-02-24")
    may_dates = pd.bdate_range("2021-02-25", "2021-02-26")

    schedule = pd.DataFrame({
        "date": april_dates.append(may_dates),
        "future_ticker": [april] * len(april_dates) + [may] * len(may_dates),
    })

    observed = schedule.loc[schedule["date"] <= valuation_date]
    future = schedule.loc[schedule["date"] > valuation_date]

    dates = pd.DatetimeIndex(future["date"]).insert(0, valuation_date)
    time_steps = (dates[1:] - dates[:-1]).days.to_numpy() / 365

    return observed, future, time_steps