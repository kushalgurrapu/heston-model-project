import numpy as np
from scipy.integrate import quad


def heston_characteristic_function(u, S0, T, r, q, kappa, theta, xi, rho, v0):
    """
    Heston characteristic function for log(S_T).
    """
    i = 1j

    x = np.log(S0)
    drift = r - q

    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi**2 * (i * u + u**2))

    g = (kappa - rho * xi * i * u - d) / (
        kappa - rho * xi * i * u + d
    )

    C = (
        i * u * (x + drift * T)
        + (kappa * theta / xi**2)
        * (
            (kappa - rho * xi * i * u - d) * T
            - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
    )

    D = (
        ((kappa - rho * xi * i * u - d) / xi**2)
        * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    )

    return np.exp(C + D * v0)


def _integrand_P1(u, S0, K, T, r, q, kappa, theta, xi, rho, v0):
    i = 1j

    phi_u_minus_i = heston_characteristic_function(
        u - i, S0, T, r, q, kappa, theta, xi, rho, v0
    )

    phi_minus_i = heston_characteristic_function(
        -i, S0, T, r, q, kappa, theta, xi, rho, v0
    )

    numerator = np.exp(-i * u * np.log(K)) * phi_u_minus_i
    denominator = i * u * phi_minus_i

    return np.real(numerator / denominator)


def _integrand_P2(u, S0, K, T, r, q, kappa, theta, xi, rho, v0):
    i = 1j

    phi_u = heston_characteristic_function(
        u, S0, T, r, q, kappa, theta, xi, rho, v0
    )

    numerator = np.exp(-i * u * np.log(K)) * phi_u
    denominator = i * u

    return np.real(numerator / denominator)


def heston_call_price(S0, K, T, r, q, kappa, theta, xi, rho, v0):
    """
    European call price under the Heston model.

    Uses:
        C = S0 * exp(-qT) * P1 - K * exp(-rT) * P2
    """

    P1_integral = quad(
        lambda u: _integrand_P1(u, S0, K, T, r, q, kappa, theta, xi, rho, v0),
        1e-8,
        100,
        limit=100,
    )[0]

    P2_integral = quad(
        lambda u: _integrand_P2(u, S0, K, T, r, q, kappa, theta, xi, rho, v0),
        1e-8,
        100,
        limit=100,
    )[0]

    P1 = 0.5 + P1_integral / np.pi
    P2 = 0.5 + P2_integral / np.pi

    price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    return max(float(price), 0.0)