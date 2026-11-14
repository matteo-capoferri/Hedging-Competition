import numpy as np

def price_collar(paths_apr21, paths_may21, Kp, Kc, DF):
    N = paths_may21.shape[1]
    A = (paths_apr21[:,:-2].sum(axis=1) + paths_may21[:,-2:].sum(axis=1))/N    
    mean_payoff = np.maximum(Kp-A, 0).mean() - np.maximum(A-Kc, 0).mean()
    return mean_payoff * DF


    #def price_collar(observed_prices, simulated_prices, Kp, Kc, DF):
    total = len(observed_prices) + simulated_prices.shape[1]
    average = (np.sum(observed_prices) + simulated_prices.sum(axis=1)) / total

    put = np.maximum(Kp - average, 0)
    call = np.maximum(average - Kc, 0)
    return DF * (put - call).mean()