import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def implied_volatility(future, strike, time, price, option_type):
    """
    INPUT:
        future = futures price per barrel.
        strike = option strike price per barrel.
        time = remaining life in years.
        price = observed option price per barrel.
        option_type = "Call" or "Put".
    OUTPUT:
        volatility = annual implied volatility as a decimal (0.20 means 20%).

    Find the volatility matching the observed price using undiscounted Black-76.
    """
    return brentq(lambda sigma: black_76(future, strike, time, sigma, option_type) - price,
                  1e-10, 5)


def black_greeks(future, strike, time, volatility, option_type):
    """
    INPUT:
        future, strike = futures price and option strike.
        time = remaining life in years.
        volatility = annual decimal implied volatility.
        option_type = "Call" or "Put".
    OUTPUT:
        delta, vega = undiscounted sensitivities per barrel.
            Vega is per unit decimal volatility; delta holds volatility fixed.
    """
    if future <= 0 or strike <= 0 or time <= 0 or volatility <= 0:
        raise ValueError("Black Greeks require positive inputs")
    if option_type not in ("Call", "Put"):
        raise ValueError("Option type must be Call or Put")
    d1 = (np.log(future / strike) + 0.5 * volatility**2 * time) / (volatility * np.sqrt(time))
    delta = norm.cdf(d1) if option_type == "Call" else norm.cdf(d1) - 1
    vega = future * norm.pdf(d1) * np.sqrt(time)
    return float(delta), float(vega)


def black_76(F, K, T, sigma, type):
    if(F<0 or K<0 or sigma<0):
        raise ValueError("valore input non valido")

# Controlli per parametri nulli
    if T == 0 or sigma == 0:
        if type == "Call":
            return max(F - K, 0)
        return max(K - F, 0)
    
    d1 = (np.log(F/K) + 0.5*T*(sigma**2))/(sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    if type == "Call":
        return F * norm.cdf(d1) - K*norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F*norm.cdf(-d1)
