from __future__ import annotations

import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor

from voiceguard.config import Config, load_config


def rho(
    d: np.ndarray,
    w: int,
    cfg: Config | None = None,
) -> float:
    """Regression predictor on relative pitch period jitter sequence (FR-10).

    Uses MLPRegressor(hidden_layer_sizes=(w,)) to predict d_{i+1} from d_{i-w+1..i}
    with continuous 80/20 train/test split.

    Returns:
        float: rho = 1 - MSE_test / Var(d_test).
    """
    if cfg is None:
        cfg = load_config()

    d_arr = np.asarray(d, dtype=np.float64)
    n = len(d_arr)
    if n <= w + 1:
        return float("nan")

    # Sliding window of length w + 1: w inputs, 1 target
    sub = np.lib.stride_tricks.sliding_window_view(d_arr, w + 1)
    X = sub[:, :w]
    y = sub[:, w]

    n_samples = len(y)
    n_train = int(n_samples * cfg.d2.train_frac)
    n_test = n_samples - n_train

    if n_test < 10 or n_train < 10:
        return float("nan")

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    var_test = float(np.var(y_test))
    if var_test <= 1e-12:
        return 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        reg = MLPRegressor(
            hidden_layer_sizes=(w,),
            activation="relu",
            solver="adam",
            learning_rate_init=cfg.d2.mlp_lr,
            random_state=cfg.d2.seed,
            max_iter=cfg.d2.max_iter,
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

    mse = float(np.mean((y_test - y_pred) ** 2))
    val_rho = 1.0 - (mse / var_test)
    return float(val_rho)
