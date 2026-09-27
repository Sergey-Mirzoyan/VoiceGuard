from __future__ import annotations

from typing import Literal

import numpy as np

from voiceguard.config import Config, load_config
from voiceguard.dsp.lpc import residual
from voiceguard.types import Segment


def binarize(e: np.ndarray, frame_len: int) -> np.ndarray:
    """Binarize LPC residual into {0, 1} bits: b[n] = 1{e[n] > median(e in frame)}."""
    e_arr = np.asarray(e, dtype=np.float32)
    n = len(e_arr)
    if n == 0 or frame_len <= 0:
        return np.zeros(0, dtype=np.uint8)

    bits = np.zeros(n, dtype=np.uint8)
    for start in range(0, n, frame_len):
        end = min(start + frame_len, n)
        frame = e_arr[start:end]
        med = float(np.median(frame))
        bits[start:end] = (frame > med).astype(np.uint8)

    return bits


def symbols_A(
    segment: Segment,
    cfg: Config | None = None,
    frames: Literal["all", "voiced", "unvoiced"] = "all",
) -> np.ndarray:
    """Extract Variant A binary symbols from segment LPC residual filtered by frame voicing."""
    if cfg is None:
        cfg = load_config()

    frame_ms = cfg.audio.frame_ms
    frame_len = max(1, int(segment.sr * frame_ms / 1000))

    e = residual(segment.samples, segment.sr, cfg=cfg)
    b = binarize(e, frame_len=frame_len)

    n_frames = len(segment.voiced_mask)
    expected_len = n_frames * frame_len

    # Truncate or pad to exactly n_frames * frame_len
    if len(b) < expected_len:
        b_padded = np.zeros(expected_len, dtype=np.uint8)
        b_padded[: len(b)] = b
        b = b_padded
    elif len(b) > expected_len:
        b = b[:expected_len]

    frame_matrix = b.reshape(n_frames, frame_len)

    if frames == "all":
        return frame_matrix.reshape(-1)
    elif frames == "voiced":
        mask = np.asarray(segment.voiced_mask, dtype=bool)
        selected = frame_matrix[mask]
        return selected.reshape(-1) if len(selected) > 0 else np.zeros(0, dtype=np.uint8)
    elif frames == "unvoiced":
        mask = ~np.asarray(segment.voiced_mask, dtype=bool)
        selected = frame_matrix[mask]
        return selected.reshape(-1) if len(selected) > 0 else np.zeros(0, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown frame class: '{frames}'. Allowed: 'all', 'voiced', 'unvoiced'")
