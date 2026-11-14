import numpy as np
from scipy.stats import norm


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