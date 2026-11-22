def central_delta(value_at, price, eps):
    """Stima il delta variando un solo prezzo del fattore relativo eps."""
    bump = eps * price
    value_up = value_at(price + bump)
    value_down = value_at(price - bump)
    return (value_up - value_down) / (2 * bump)
