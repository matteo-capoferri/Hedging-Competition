import numpy as np


def check_accounting(result):
    """Controlla la riconciliazione giornaliera e restituisce l'errore in USD."""
    # Controllo condiviso da esecuzioni reali e test: non cambia il P&L.
    pnl_from_values = result["portfolio_value"].diff()
    pnl_from_values.iloc[0] = result["portfolio_value"].iloc[0]
    error = result["delta_hedged_pnl"] - pnl_from_values
    if not np.isfinite(error).all() or not np.allclose(error, 0.0, rtol=0.0, atol=1e-6):
        raise ValueError("Il P&L per componenti non riconcilia con il valore del portafoglio.")
    return error
