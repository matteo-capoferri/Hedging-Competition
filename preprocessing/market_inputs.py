def market_inputs(today, contract, active, spot, smiles):
    """Legge il prezzo e lo smile del future se ha ancora fixing residui."""
    if active:
        return spot.at[(today, contract), "settlement"], smiles[contract][today]
    # Il simulatore richiede due futures. I percorsi di quello inattivo sono scartati.
    return 1.0, {"K": [1.0], "IV": [0.0]}
