import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from heston.market import S0, K_grid, T_grid
from heston.sampling import sample_heston_params, sample_market_params
from heston.pricing import heston_call_price

#Makes one set of samples, 5*4=20 prices, 2 market parameters, 5 heston parameters, everything is normalized
def generate_one_sample():
    heston_params = sample_heston_params()
    market_params = sample_market_params()

    kappa, theta, xi, rho, v0 = heston_params
    r, q = market_params
    r_scaled = (r - 0.025) / (0.055 - 0.025)
    q_scaled = (q - 0.005) / (0.025 - 0.005)
    market_scaled = np.array([r_scaled, q_scaled])
    prices = []

    for T in T_grid:
        for K in K_grid:
            price = heston_call_price(
                S0, K, T, r, q,
                kappa, theta, xi, rho, v0
            )
            prices.append(price / S0)

    prices = np.array(prices)

    X = np.concatenate([prices, market_scaled])
    y = heston_params
    y_min = np.array([0.5, 0.005, 0.1, -0.95, 0.005])
    y_max = np.array([4.0, 0.09, 0.8, -0.3, 0.12])

    y_scaled = (y - y_min) / (y_max - y_min)

    return X, y_scaled

def generate_dataset(N):
    X_data = []
    y_data = []

    for _ in tqdm(range(N)):
        X, y = generate_one_sample()  # ← just call it directly
        X_data.append(X)
        y_data.append(y)

    return np.array(X_data), np.array(y_data)

def _worker(_):
    return generate_one_sample()

def generate_dataset_parallel(N, n_workers=None, chunksize=1):
    if n_workers is None:
        n_workers = cpu_count()

    X_data = []
    y_data = []
    with Pool(n_workers) as pool:
        for X, y in tqdm(pool.imap_unordered(_worker, range(N), chunksize=chunksize), total=N):
            X_data.append(X)
            y_data.append(y)

    return np.array(X_data), np.array(y_data)