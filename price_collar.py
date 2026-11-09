import numpy as np

def price_collar(paths_apr21, paths_may21, Kp, Kc, DF):
    N = paths_may21.shape[1]
    A = (paths_apr21[:,:-2].sum(axis=1) + paths_may21[:,-2:].sum(axis=1))/N    
    mean_payoff = np.maximum(Kp-A, 0).mean() + np.maximum(A-Kc, 0).mean()
    return mean_payoff * DF
