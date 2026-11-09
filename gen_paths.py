import numpy as np

def gen_paths(M, n_step, rho, sigma_1, sigma_2, dt, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, n_step, 2))
    #creiamo W correlati w1 = z1    w2=z1*rho + z2*sqrt(1-rho**2)
    W1 = Z[:,:,0]
    W2 = W1*rho + Z[:,:,1] * np.sqrt(1-rho**2)


    increments_1 = (-0.5 * sigma_1**2 * dt + np.sqrt(dt) * W1)
    log_returns_1 = np.cumsum(increments_1, axis=1)

    increments_2 = (-0.5 * sigma_2**2 * dt + np.sqrt(dt) * W1)
    log_returns_2 = np.cumsum(increments_2, axis=1)
    return [log_returns_1, log_returns_2]