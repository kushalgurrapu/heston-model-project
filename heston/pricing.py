import numpy as np
from scipy.integrate import quad

def heston_call_price(S0, K, T, r, q, kappa, theta, xi, rho, v0,
                      umax=250, N=2500):
    I = 1j
    P = 0
    du = umax / N

    aa = theta * kappa * T / xi**2
    bb = -2 * theta * kappa / xi**2

    for i in range(1, N):
        u2 = i * du
        u1 = complex(u2, -1)

        a1 = rho * xi * u1 * I
        a2 = rho * xi * u2 * I

        d1 = np.sqrt((a1 - kappa) ** 2 + xi**2 * (u1 * I + u1**2))
        d2 = np.sqrt((a2 - kappa) ** 2 + xi**2 * (u2 * I + u2**2))

        g1 = (kappa - a1 - d1) / (kappa - a1 + d1)
        g2 = (kappa - a2 - d2) / (kappa - a2 + d2)

        b1 = np.exp(u1 * I * (np.log(S0 / K) + (r - q) * T)) * (
            (1 - g1 * np.exp(-d1 * T)) / (1 - g1)
        ) ** bb

        b2 = np.exp(u2 * I * (np.log(S0 / K) + (r - q) * T)) * (
            (1 - g2 * np.exp(-d2 * T)) / (1 - g2)
        ) ** bb

        phi1 = b1 * np.exp(
            aa * (kappa - a1 - d1)
            + v0
            * (kappa - a1 - d1)
            * (1 - np.exp(-d1 * T))
            / (1 - g1 * np.exp(-d1 * T))
            / xi**2
        )

        phi2 = b2 * np.exp(
            aa * (kappa - a2 - d2)
            + v0
            * (kappa - a2 - d2)
            * (1 - np.exp(-d2 * T))
            / (1 - g2 * np.exp(-d2 * T))
            / xi**2
        )

        P += ((phi1 - phi2) / (u2 * I)) * du

    return K * np.real((S0 / K - np.exp(-r * T)) / 2 + P / np.pi)