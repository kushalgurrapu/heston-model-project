import numpy as np

S0 = 100.0

moneyness_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
T_grid = np.array([0.25, 0.5, 1.0, 2.0])

K_grid = S0 * moneyness_grid