from heston.sampling import *
from heston.market import S0, T_grid, K_grid
from heston.pricing import heston_call_price
from heston.dataset import generate_one_sample, generate_dataset

print("Market Settings:")
print("S0 = ", S0)
print("K_grid = ", K_grid)
print("T_grid = ", T_grid)

N = 1000
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
X, y = generate_dataset(N)

print("Done!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Save dataset
np.save("data/X_1000.npy", X)
np.save("data/y_1000.npy", y)

print("Saved to data/X_1000.npy and data/y_1000.npy")