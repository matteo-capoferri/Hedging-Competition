import numpy as np


def read_fixings(observed, spot):
    """Legge i settlement dei fixing già avvenuti, in ordine di calendario."""
    observed_prices = np.array([
        spot.at[(row.date, row.future_ticker), "settlement"]
        for row in observed.itertuples(index=False)
    ])
    if not np.all(np.isfinite(observed_prices)):
        raise ValueError("Observed fixings must be finite")
    return observed_prices
