import argparse
import os

import numpy as np

from heston.market import S0, T_grid, K_grid
from heston.dataset import generate_dataset_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    args = parser.parse_args()
    N = args.N

    print("Market Settings:")
    print("S0 = ", S0)
    print("K_grid = ", K_grid)
    print("T_grid = ", T_grid)

    # Test if you change N and umax vals
    # price = heston_call_price(
    #     S0=95,
    #     K=100,
    #     T=2,
    #     r=0.03,
    #     q=0.0,
    #     kappa=1.5768,
    #     theta=0.0398,
    #     xi=0.575,
    #     rho=-0.5711,
    #     v0=0.1
    # )
    # print(price)
    # Should be around 12.356

    print(f"Generating {N} samples...")
    X, y = generate_dataset_parallel(N)

    print("Done!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save the dataset
    os.makedirs("data", exist_ok=True)
    x_path = f"data/X_{N}.npy"
    y_path = f"data/y_{N}.npy"
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"Saved to {x_path} and {y_path}")