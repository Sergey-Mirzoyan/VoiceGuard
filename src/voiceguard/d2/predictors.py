from __future__ import annotations

import warnings
from typing import Any

import joblib
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from voiceguard.config import Config, load_config
from voiceguard.d2.regression import rho


def bit_delta(
    bits: np.ndarray,
    w: int,
    direction: str,
    cfg: Config | None = None,
    max_train: int | None = None,
) -> tuple[float, int]:
    """Bitwise prediction mode (FR-01).

    Args:
        bits: Binary 0/1 sequence.
        w: Window size (context length).
        direction: 'fwd' (forward) or 'bwd' (backward, reversed sequence).
        cfg: Configuration object.
        max_train: Optional override for maximum training samples.

    Returns:
        tuple[float, int]: (delta, N_test). delta = accuracy - 0.5.
    """
    if cfg is None:
        cfg = load_config()

    b_arr = np.asarray(bits, dtype=np.uint8)
    if direction == "bwd":
        b_arr = b_arr[::-1]

    n = len(b_arr)
    if n <= w:
        return float("nan"), 0

    sub = np.lib.stride_tricks.sliding_window_view(b_arr, w + 1)
    X = sub[:, :w].astype(np.float32)
    y = sub[:, w].astype(np.int64)

    n_samples = len(y)
    n_train_total = int(n_samples * cfg.d2.train_frac)
    n_test = n_samples - n_train_total

    # Per FR-03: fewer than 200 test symbols returns NaN
    if n_test < 200 or n_train_total == 0:
        return float("nan"), n_test

    t_limit = max_train if max_train is not None else cfg.d2.max_train
    t_actual = min(n_train_total, t_limit)

    X_train = X[n_train_total - t_actual : n_train_total]
    y_train = y[n_train_total - t_actual : n_train_total]
    X_test = X[n_train_total:]
    y_test = y[n_train_total:]

    # Degenerate case: single class in training set
    classes = np.unique(y_train)
    if len(classes) < 2:
        acc = float(np.mean(y_test == classes[0]))
        return float(acc - 0.5), n_test

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf = MLPClassifier(
            hidden_layer_sizes=(w,),
            activation="relu",
            solver="adam",
            learning_rate_init=cfg.d2.mlp_lr,
            random_state=cfg.d2.seed,
            max_iter=cfg.d2.max_iter,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = float(np.mean(y_pred == y_test))
    return float(acc - 0.5), n_test


def block_delta(
    bits: np.ndarray,
    w: int,
    direction: str,
    cfg: Config | None = None,
    max_train: int | None = None,
) -> tuple[float, int]:
    """Blockwise prediction mode (FR-02).

    Sequence is divided into non-overlapping blocks of w bits.
    Pairs (B_i, B_{i+1}) are used to predict block B_{i+1} from B_i.
    StandardScaler is fitted only on the training portion.
    Prediction output is thresholded at 0.5.

    Returns:
        tuple[float, int]: (delta, N_test_bits). delta = accuracy - 0.5.
    """
    if cfg is None:
        cfg = load_config()

    b_arr = np.asarray(bits, dtype=np.uint8)
    if direction == "bwd":
        b_arr = b_arr[::-1]

    n_blocks = len(b_arr) // w
    if n_blocks < 2:
        return float("nan"), 0

    blocks = b_arr[: n_blocks * w].reshape(n_blocks, w).astype(np.float32)
    X_b = blocks[:-1]
    Y_b = blocks[1:]

    n_pairs = len(Y_b)
    n_train_total = int(n_pairs * cfg.d2.train_frac)
    n_test_blocks = n_pairs - n_train_total
    n_test_bits = n_test_blocks * w

    if n_test_bits < 200 or n_train_total == 0:
        return float("nan"), n_test_bits

    t_limit = max_train if max_train is not None else cfg.d2.max_train
    t_actual = min(n_train_total, t_limit)

    X_train = X_b[n_train_total - t_actual : n_train_total]
    Y_train = Y_b[n_train_total - t_actual : n_train_total]
    X_test = X_b[n_train_total:]
    Y_test = Y_b[n_train_total:]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = MLPRegressor(
            hidden_layer_sizes=(w,),
            activation="relu",
            solver="adam",
            learning_rate_init=cfg.d2.mlp_lr,
            random_state=cfg.d2.seed,
            max_iter=cfg.d2.max_iter,
        )
        reg.fit(X_train_s, Y_train)
        preds = (reg.predict(X_test_s) >= 0.5).astype(np.uint8)

    acc = float(np.mean(preds == Y_test))
    return float(acc - 0.5), n_test_bits


def _execute_check(
    check_id: str,
    bits: np.ndarray,
    mode: str,
    w: int,
    direction: str,
    cfg: Config,
    max_train: int | None = None,
) -> tuple[str, float, int]:
    if mode == "bit":
        delta, n_test = bit_delta(bits, w=w, direction=direction, cfg=cfg, max_train=max_train)
    elif mode == "block":
        delta, n_test = block_delta(bits, w=w, direction=direction, cfg=cfg, max_train=max_train)
    else:
        raise ValueError(f"Unknown check mode: {mode}")
    return check_id, delta, n_test


def run_checks(
    bits: np.ndarray,
    variant: str = "A",
    cfg: Config | None = None,
    d_seq: np.ndarray | None = None,
) -> dict[str, tuple[float, int]]:
    """Execute all configured predictor checks in parallel (FR-03).

    Args:
        bits: Binary symbol array.
        variant: "A" (LPC residual bits) or "B" (pitch jitter bits).
        cfg: Configuration.
        d_seq: Relative jitter sequence (for Variant B regression check rho).

    Returns:
        dict[str, tuple[float, int]]: Mapping check_id -> (delta, N_test).
    """
    if cfg is None:
        cfg = load_config()

    windows = cfg.d2.windows_B if variant == "B" else cfg.d2.windows
    tasks: list[tuple[str, str, int, str]] = []

    for mode in cfg.d2.modes:
        for w in windows:
            for direction in cfg.d2.directions:
                check_id = f"{variant}_{mode}_w{w}_{direction}"
                tasks.append((check_id, mode, w, direction))

    results_list = joblib.Parallel(n_jobs=cfg.d2.n_jobs)(
        joblib.delayed(_execute_check)(cid, bits, mode, w, direction, cfg)
        for cid, mode, w, direction in tasks
    )

    checks: dict[str, tuple[float, int]] = {
        cid: (delta, n_test) for cid, delta, n_test in results_list
    }

    # Additional regression checks for Variant B (FR-10)
    if variant == "B" and d_seq is not None and len(d_seq) > 0:
        for w in windows:
            rho_id = f"B_rho_w{w}_fwd"
            rho_val = rho(d_seq, w=w, cfg=cfg)
            n_samples = max(0, len(d_seq) - w)
            n_test = int(n_samples * (1.0 - cfg.d2.train_frac))
            checks[rho_id] = (rho_val, n_test)

    return checks


def dissertation_verdict(
    bits: np.ndarray,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Compatibility mode with the dissertation procedure for PRNG testing (FR-04).

    Training size T = min(floor(n / 5), 50), 80/20 split.
    Overall delta = max delta over 12 checks.
    Threshold: eps = z_{alpha / (2m)} / (2 * sqrt(N_test)), m = 12.
    Verdict: 'нарушение ТСБ' if delta_max > eps.
    """
    if cfg is None:
        cfg = load_config()

    b_arr = np.asarray(bits, dtype=np.uint8)
    n = len(b_arr)
    t_train = min(n // 5, 50)
    m = 12

    windows = cfg.d2.windows
    tasks = []
    for mode in cfg.d2.modes:
        for w in windows:
            for direction in cfg.d2.directions:
                check_id = f"A_{mode}_w{w}_{direction}"
                tasks.append((check_id, mode, w, direction))

    results_list = joblib.Parallel(n_jobs=cfg.d2.n_jobs)(
        joblib.delayed(_execute_check)(cid, b_arr, mode, w, direction, cfg, max_train=t_train)
        for cid, mode, w, direction in tasks
    )

    deltas: dict[str, float] = {}
    n_test_total = 0
    for cid, delta, n_test in results_list:
        deltas[cid] = delta
        if n_test > 0:
            n_test_total = n_test

    valid_deltas = [d for d in deltas.values() if not np.isnan(d)]
    if not valid_deltas or n_test_total == 0:
        return {
            "delta_max": float("nan"),
            "eps": float("nan"),
            "is_violation": False,
            "verdict": "недостаточно данных",
            "deltas": deltas,
            "n_test": n_test_total,
        }

    delta_max = float(max(valid_deltas))
    z = float(stats.norm.isf(cfg.d2.alpha / (2.0 * m)))
    eps = float(z / (2.0 * np.sqrt(n_test_total)))
    is_violation = bool(delta_max > eps)

    return {
        "delta_max": delta_max,
        "eps": eps,
        "is_violation": is_violation,
        "verdict": "нарушение ТСБ" if is_violation else "соответствие ТСБ",
        "deltas": deltas,
        "n_test": n_test_total,
    }
