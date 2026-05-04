import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from heston.model import BUILDERS

# Mirrors heston/dataset.py:32-33 — kept in sync manually.
PARAM_NAMES = ['kappa', 'theta', 'xi', 'rho', 'v0']
Y_MIN = np.array([0.5, 0.005, 0.1, -0.95, 0.005])
Y_MAX = np.array([4.0, 0.09, 0.8, -0.3, 0.12])

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def load_dataset(n_samples):
    x_path = DATA_DIR / f'X_{n_samples}.npy'
    y_path = DATA_DIR / f'y_{n_samples}.npy'
    if not x_path.exists() or not y_path.exists():
        sys.exit(
            f"Missing {x_path} or {y_path}. Run `python main.py --N {n_samples}` first to generate the dataset."
        )
    return np.load(x_path), np.load(y_path)


def unscale(y_scaled):
    return y_scaled * (Y_MAX - Y_MIN) + Y_MIN


def per_param_metrics(y_true_scaled, y_pred_scaled):
    y_true = unscale(y_true_scaled)
    y_pred = unscale(y_pred_scaled)
    abs_err = np.abs(y_pred - y_true)
    mae = abs_err.mean(axis=0)
    rel = (abs_err / np.abs(y_true)).mean(axis=0)
    return {
        name: {'mae': float(mae[i]), 'mean_relative_error': float(rel[i])}
        for i, name in enumerate(PARAM_NAMES)
    }


def plot_loss(history, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss (scaled targets)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(BUILDERS), default='modern')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    X, y = load_dataset(args.n_samples)
    print(f"Loaded X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.seed
    )

    model = BUILDERS[args.variant](input_dim=X.shape[1], output_dim=y.shape[1])
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss', mode='min', patience=40,
        min_delta=1e-7, restore_best_weights=True, verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    test_mse = float(model.evaluate(X_test, y_test, verbose=0))
    y_pred = model.predict(X_test, verbose=0)
    metrics = {
        'variant': args.variant,
        'test_mse_scaled': test_mse,
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'epochs_run': len(history.history['loss']),
        'per_param': per_param_metrics(y_test, y_pred),
    }

    out_dir = MODELS_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / 'model.keras')
    with open(out_dir / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    plot_loss(history.history, out_dir / 'loss_curve.png')

    print(f"\nTest MSE (scaled targets): {test_mse:.6f}")
    print("Per-parameter errors (original units):")
    for name, m in metrics['per_param'].items():
        print(f"  {name:6s}  MAE={m['mae']:.5f}  rel={m['mean_relative_error']*100:.2f}%")
    print(f"\nArtifacts written to {out_dir}/")


if __name__ == '__main__':
    main()
