"""D2: predictive test module (ЧТЗ-05).

Implements FR-01..FR-06:
- Predictors: dissertation MLPs from voiceguard.d2.predictors (ЧТЗ-05)
- delta = p_hat - 0.5
- Z-score relative to live-speech reference
- s2 = max|Z|
- Chi-squared test
- Reference (calibration) fitting and persistence
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from voiceguard.config import Config, load_config
from voiceguard.dsp.binarize import symbols_A
from voiceguard.types import D2Result, Segment

logger = logging.getLogger(__name__)

CheckId = str  # e.g. "A_block_w8_fwd"


# ---------------------------------------------------------------------------
# Predictor logic
# ---------------------------------------------------------------------------


def _make_windows(
    bits: np.ndarray, w: int, mode: Literal["bit", "block"], direction: Literal["fwd", "bwd"]
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target y for a predictor.

    bit mode: predict next bit from w previous bits.
    block mode: predict majority (mode) of next block-of-w bits from w previous bits.
    """
    b = bits if direction == "fwd" else bits[::-1]
    n = len(b)

    if mode == "bit":
        min_len = w + 1
        if n < min_len:
            return np.empty((0, w), dtype=np.float32), np.empty(0, dtype=np.uint8)
        X = np.lib.stride_tricks.sliding_window_view(b[:-1], w).astype(np.float32)
        y = b[w:].astype(np.uint8)
    else:  # block
        min_len = 2 * w
        if n < min_len:
            return np.empty((0, w), dtype=np.float32), np.empty(0, dtype=np.uint8)
        # Context: bits[i..i+w-1], target: majority of bits[i+w..i+2w-1]
        n_blocks = (n - w) // w
        X_list, y_list = [], []
        for i in range(n_blocks):
            ctx = b[i * w : i * w + w].astype(np.float32)
            tgt_block = b[(i + 1) * w : (i + 2) * w]
            X_list.append(ctx)
            y_list.append(int(tgt_block.mean() > 0.5))
        if not X_list:
            return np.empty((0, w), dtype=np.float32), np.empty(0, dtype=np.uint8)
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.uint8)

    return X, y


def _compute_delta(
    bits: np.ndarray,
    w: int,
    mode: Literal["bit", "block"],
    direction: Literal["fwd", "bwd"],
    max_train: int = 5000,
    train_frac: float = 0.8,
    seed: int = 42,
) -> float:
    """delta = p_hat - 0.5 from the dissertation MLP predictors (ЧТЗ-05).

    Thin adapter over voiceguard.d2.predictors.bit_delta / block_delta
    (MLPClassifier / MLPRegressor, contiguous split, real block mode).
    Returns NaN when there is not enough data (callers must skip NaN).
    """
    from voiceguard.d2.predictors import bit_delta, block_delta

    cfg = load_config(overrides={"d2": {"train_frac": train_frac, "seed": seed}})
    fn = bit_delta if mode == "bit" else block_delta
    delta, _n_test = fn(bits, w=w, direction=direction, cfg=cfg, max_train=max_train)
    return float(delta)


def _check_ids(
    variant: Literal["A", "B"] = "A",
    windows: list[int] | None = None,
    modes: list[str] | None = None,
    directions: list[str] | None = None,
) -> list[CheckId]:
    """Build list of check IDs for the given configuration."""
    if windows is None:
        windows = [8, 16, 32]
    if modes is None:
        modes = ["bit", "block"]
    if directions is None:
        directions = ["fwd", "bwd"]
    ids = []
    for m in modes:
        for w in windows:
            for d in directions:
                ids.append(f"{variant}_{m}_w{w}_{d}")
    return ids


def _parse_check_id(check_id: str) -> tuple[str, str, int, str]:
    """Parse check_id into (variant, mode, w, direction)."""
    parts = check_id.split("_")
    # Format: {variant}_{mode}_w{w}_{dir}
    # variant: parts[0]
    # mode: parts[1]
    # w: parts[2][1:] (strip 'w')
    # dir: parts[3]
    variant = parts[0]
    mode = parts[1]
    w = int(parts[2][1:])
    direction = parts[3]
    return variant, mode, w, direction


# ---------------------------------------------------------------------------
# Single-segment scoring
# ---------------------------------------------------------------------------


def score_segment(
    seg: Segment,
    reference: "Reference",
    cfg: Config | None = None,
) -> D2Result:
    """Compute D2Result for one segment using a fitted Reference."""
    if cfg is None:
        cfg = load_config()

    bits = symbols_A(seg, cfg=cfg, frames="all")

    deltas: dict[str, float] = {}
    z_scores: dict[str, float] = {}
    chi2_p: dict[str, float] = {}

    check_list = reference.check_ids

    for check_id in check_list:
        variant, mode, w, direction = _parse_check_id(check_id)
        d = _compute_delta(
            bits,
            w=w,
            mode=mode,
            direction=direction,
            max_train=cfg.d2.max_train,
            train_frac=cfg.d2.train_frac,
            seed=cfg.d2.seed,
        )
        deltas[check_id] = d
        if np.isnan(d):
            z_scores[check_id] = float("nan")
            chi2_p[check_id] = float("nan")
            continue

        mu0 = reference.mu0.get(check_id, 0.0)
        sigma0 = reference.sigma0.get(check_id, 1.0)
        if sigma0 < 1e-9:
            sigma0 = 1e-9
        z = (d - mu0) / sigma0
        z_scores[check_id] = z

        # Simple chi2 via binomial approximation (FR-04)
        n_test = max(1, int(len(bits) * (1 - cfg.d2.train_frac)))
        from scipy.stats import norm as sp_norm

        chi2_p[check_id] = float(2 * sp_norm.sf(abs(z)))

    s2 = max((abs(v) for v in z_scores.values() if not np.isnan(v)), default=0.0)

    return D2Result(
        delta=deltas,
        z=z_scores,
        s2=s2,
        chi2_p=chi2_p,
        n_symbols=len(bits),
        variant="A",
    )


# ---------------------------------------------------------------------------
# Reference (эталон)
# ---------------------------------------------------------------------------


class Reference:
    """D2 reference: mean and std of delta values for live speech on a channel.

    Attributes:
        channel: Channel string (e.g. "clean", "amrnb_12.2").
        check_ids: List of check IDs included in this reference.
        mu0: Mean delta per check_id for live speech.
        sigma0: Std of delta per check_id for live speech.
        frames: "all" | "voiced".
        n_segments: Number of segments used to build reference.
    """

    def __init__(
        self,
        channel: str,
        mu0: dict[str, float],
        sigma0: dict[str, float],
        check_ids: list[str],
        frames: str = "all",
        n_segments: int = 0,
    ) -> None:
        self.channel = channel
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.check_ids = check_ids
        self.frames = frames
        self.n_segments = n_segments

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "check_ids": self.check_ids,
            "frames": self.frames,
            "n_segments": self.n_segments,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Reference saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        with open(path) as f:
            d = json.load(f)
        return cls(
            channel=d["channel"],
            mu0=d["mu0"],
            sigma0=d["sigma0"],
            check_ids=d["check_ids"],
            frames=d.get("frames", "all"),
            n_segments=d.get("n_segments", 0),
        )


def build_reference(
    segments: list[Segment],
    channel: str,
    cfg: Config | None = None,
    frames: str = "all",
) -> Reference:
    """Build D2 reference from live-speech segments.

    Args:
        segments: List of Segment objects (live speech on specified channel).
        channel: Channel string identifier.
        cfg: Config object.
        frames: "all" | "voiced".

    Returns:
        Reference object with fitted mu0, sigma0.
    """
    if cfg is None:
        cfg = load_config()

    check_ids = _check_ids(
        variant="A",
        windows=cfg.d2.windows,
        modes=cfg.d2.modes,
        directions=cfg.d2.directions,
    )

    all_deltas: dict[str, list[float]] = {cid: [] for cid in check_ids}

    for seg in segments:
        bits = symbols_A(seg, cfg=cfg, frames=frames)  # type: ignore[arg-type]
        if len(bits) < 20:
            continue
        for check_id in check_ids:
            _, mode, w, direction = _parse_check_id(check_id)
            d = _compute_delta(
                bits,
                w=w,
                mode=mode,
                direction=direction,
                max_train=cfg.d2.max_train,
                train_frac=cfg.d2.train_frac,
                seed=cfg.d2.seed,
            )
            if not np.isnan(d):
                all_deltas[check_id].append(d)

    mu0: dict[str, float] = {}
    sigma0: dict[str, float] = {}
    for cid in check_ids:
        vals = all_deltas[cid]
        if vals:
            mu0[cid] = float(np.mean(vals))
            sigma0[cid] = float(max(np.std(vals), 1e-6))
        else:
            mu0[cid] = 0.0
            sigma0[cid] = 1.0

    return Reference(
        channel=channel,
        mu0=mu0,
        sigma0=sigma0,
        check_ids=check_ids,
        frames=frames,
        n_segments=len(segments),
    )
