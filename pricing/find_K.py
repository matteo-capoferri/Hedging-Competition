from scipy.optimize import brentq
import numpy as np

def find_K(paths_apr21, paths_may21, Kc):
#we need to impose that call(Kc)=put(Kp)
    N = paths_may21.shape[1]
    A = (paths_apr21[:,:-2].sum(axis=1) + paths_may21[:,-2:].sum(axis=1))/N
    K = brentq(lambda x: np.maximum(x - A, 0).mean() - np.maximum(A - Kc, 0).mean(),1e-5,150)
    return K
