import logging


def market_inputs(today, contract, active, spot, smiles):
    """Legge il prezzo e lo smile del future se ha ancora fixing residui."""
    if not active:
        # Il simulatore scarta i percorsi del contratto senza fixing residui.
        return 1.0, {"K": [1.0], "IV": [0.0]}

    price = spot.at[(today, contract), "settlement"]
    history = smiles.get(contract, {})
    if today in history:
        return price, history[today]

    previous_dates = [day for day in history if day < today]
    if not previous_dates:
        raise ValueError(f"Nessuno smile disponibile per {contract} al {today:%Y-%m-%d}")
    previous_day = max(previous_dates)
    previous_smile = history[previous_day]
    previous_price = spot.at[(previous_day, contract), "settlement"]

    # Mantiene la IV a parità di moneyness K/F.
    scale = price / previous_price
    strikes = [strike * scale for strike in previous_smile["K"]]
    logging.getLogger(__name__).warning(
        "%s il %s: uso lo smile del %s", contract, today.date(), previous_day.date()
    )
    return price, {"K": strikes, "IV": previous_smile["IV"]}
