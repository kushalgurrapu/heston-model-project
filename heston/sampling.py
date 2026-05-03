import numpy as np


def sample_heston_params():
    while True:
        #ranges for the heston parameters
        kappa = np.random.uniform(0.5, 4.0)
        theta = np.random.uniform(0.005, 0.09)
        xi = np.random.uniform(0.1, 0.8)
        rho = np.random.uniform(-0.95, -0.30)
        v0 = np.random.uniform(0.005, 0.12)

        #feller condition
        #basically makes sure variance (v_t) doesn't go to 0 too fast
        if 2 * kappa * theta >= xi**2:
            return np.array([kappa, theta, xi, rho, v0])

def sample_market_params():
    r = np.random.uniform(0.025, 0.055)
    q = np.random.uniform(0.005, 0.025)
    return np.array([r, q])