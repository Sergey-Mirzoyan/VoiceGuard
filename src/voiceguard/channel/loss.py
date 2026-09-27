from __future__ import annotations

import numpy as np


def generate_loss_mask(
    n_frames: int,
    loss_pct: float,
    rng: np.random.Generator,
    model: str = "independent",
    p: float = 0.05,
    r: float = 0.5,
) -> np.ndarray:
    """Generate boolean frame loss mask of length n_frames (True = lost)."""
    if n_frames <= 0 or loss_pct <= 0.0:
        return np.zeros(max(0, n_frames), dtype=bool)

    if loss_pct >= 100.0:
        return np.ones(n_frames, dtype=bool)

    if model == "gilbert_elliott":
        # Target average loss rate L in (0, 1)
        target_l = loss_pct / 100.0
        # If r is given, transition probability p is adjusted so stationary loss = target_l:
        # L = p / (p + r) => p = (target_l * r) / (1 - target_l)
        if 0.0 < target_l < 1.0 and r > 0.0:
            eff_p = float(np.clip((target_l * r) / (1.0 - target_l), 1e-6, 1.0))
            eff_r = float(np.clip(r, 1e-6, 1.0))
        else:
            eff_p = float(np.clip(p, 1e-6, 1.0))
            eff_r = float(np.clip(r, 1e-6, 1.0))

        mask = np.zeros(n_frames, dtype=bool)
        state = 1 if rng.uniform(0.0, 1.0) < target_l else 0
        mask[0] = bool(state)

        for i in range(1, n_frames):
            if state == 0:
                if rng.uniform(0.0, 1.0) < eff_p:
                    state = 1
            else:
                if rng.uniform(0.0, 1.0) < eff_r:
                    state = 0
            mask[i] = bool(state)
        return mask

    # Default: independent Bernoulli trials
    return rng.uniform(0.0, 100.0, size=n_frames) < loss_pct


def apply_pcm_plc(
    samples: np.ndarray,
    frame_len: int,
    loss_mask: np.ndarray,
    attenuation_db: float = 6.0,
) -> np.ndarray:
    """Apply Packet Loss Concealment (PLC) on PCM samples by frame repetition with attenuation."""
    if len(loss_mask) == 0 or not np.any(loss_mask):
        return samples.copy()

    out = samples.copy()
    consecutive_lost = 0
    last_good_frame = np.zeros(frame_len, dtype=samples.dtype)
    fade_len = min(16, frame_len // 4)

    for i, is_lost in enumerate(loss_mask):
        start = i * frame_len
        end = min(start + frame_len, len(samples))
        actual_frame_len = end - start

        if is_lost:
            consecutive_lost += 1
            att_factor = float(10.0 ** (-attenuation_db * consecutive_lost / 20.0))
            concealed = last_good_frame[:actual_frame_len] * att_factor

            # Boundary cross-fade if previous frame was good to avoid clicks
            if consecutive_lost == 1 and start > 0 and fade_len > 0:
                prev_end = start
                prev_start = max(0, prev_end - fade_len)
                actual_fade = prev_end - prev_start
                if actual_fade > 0:
                    alpha = np.linspace(0.0, 1.0, actual_fade, dtype=samples.dtype)
                    out[start : start + actual_fade] = (1.0 - alpha) * out[
                        start - actual_fade : start
                    ] + alpha * concealed[:actual_fade]
                    out[start + actual_fade : end] = concealed[actual_fade:]
                else:
                    out[start:end] = concealed
            else:
                out[start:end] = concealed
        else:
            consecutive_lost = 0
            last_good_frame[:actual_frame_len] = out[start:end]

    return out
