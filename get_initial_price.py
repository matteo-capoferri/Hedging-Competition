def get_initial_price(options):
    """
    INPUT:
        options = quotes for one futures contract on the selected market date.
            underlying_settlement = futures settlement price.

    OUTPUT:
        initial_price = the single available futures settlement price, as a number.

    Check that the price is available and unique, then return it.
    """
    prices = options["underlying_settlement"].dropna().unique()
    if len(prices) != 1:
        raise ValueError("The initial futures price must be available and unique")
    return float(prices[0])
