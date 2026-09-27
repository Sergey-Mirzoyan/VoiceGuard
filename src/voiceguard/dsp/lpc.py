from __future__ import annotations

import numpy as np
from scipy import signal

from voiceguard.config import Config, load_config


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Levinson-Durbin algorithm solving Yule-Walker equations.

    Returns:
        tuple[np.ndarray, float]: filter coefficients a[1..p] and prediction error variance.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    if len(r_arr) < order + 1:
        raise ValueError(f"r must have length >= order + 1 ({order + 1}), got {len(r_arr)}")

    # Regularization per FR-02 to prevent degeneration on silence
    r0 = r_arr[0] * (1.0 + 1e-9)
    if r0 <= 0.0:
        return np.zeros(order, dtype=np.float64), 0.0

    a = np.zeros(order, dtype=np.float64)
    error = r0

    for i in range(order):
        acc = r_arr[i + 1]
        for j in range(i):
            acc -= a[j] * r_arr[i - j]

        k = acc / error
        a_new = a.copy()
        a_new[i] = k
        for j in range(i):
            a_new[j] = a[j] - k * a[i - 1 - j]
        a = a_new
        error *= 1.0 - k * k
        if error <= 0.0:
            break

    return a, float(error)


def lpc_frame(
    frame: np.ndarray,
    order: int,
    window: str = "hamming",
) -> np.ndarray:
    """Estimate LPC predictor coefficients a[1..p] for a single audio frame."""
    frame_arr = np.asarray(frame, dtype=np.float64)
    n = len(frame_arr)
    if n <= order:
        return np.zeros(order, dtype=np.float64)

    try:
        w = signal.windows.get_window(window, n)
    except Exception:
        w = np.hamming(n)

    windowed = frame_arr * w
    r = signal.correlate(windowed, windowed, mode="full")[n - 1 : n + order]
    a, _ = levinson_durbin(r, order)
    return a


def residual(
    samples: np.ndarray,
    sr: int,
    cfg: Config | None = None,
) -> np.ndarray:
    """Compute LPC residual e[n] = x[n] - sum(a_k * x[n-k]) via frame-by-frame inverse filtering."""
    if cfg is None:
        cfg = load_config()

    order = cfg.lpc.order_wb if sr >= 16000 else cfg.lpc.order_nb
    frame_ms = cfg.audio.frame_ms
    frame_len = max(order + 1, int(sr * frame_ms / 1000))
    window = cfg.lpc.window

    x = np.asarray(samples, dtype=np.float32)
    total_len = len(x)
    if total_len == 0:
        return np.zeros(0, dtype=np.float32)

    e = np.zeros(total_len, dtype=np.float32)
    state = np.zeros(order, dtype=np.float64)

    for start in range(0, total_len, frame_len):
        end = min(start + frame_len, total_len)
        cur_frame = x[start:end]
        cur_len = len(cur_frame)

        if cur_len <= order:
            # Not enough samples for LPC estimation, keep state
            e[start:end] = cur_frame
            continue

        a = lpc_frame(cur_frame, order=order, window=window)

        # Extended input with previous state prepended
        x_ext = np.concatenate([state, cur_frame.astype(np.float64)])
        # Compute e[n] = x[n] - sum_{j=0}^{order-1} a[j] * x[n-1-j]
        pred = np.zeros(cur_len, dtype=np.float64)
        for j in range(order):
            pred += a[j] * x_ext[order - 1 - j : order - 1 - j + cur_len]

        e_frame = cur_frame.astype(np.float64) - pred
        e[start:end] = e_frame.astype(np.float32)

        # Update carry-over state with last p input samples of this frame
        state = x_ext[-order:].copy()

    return e
