from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voiceguard.config import Config, load_config
from voiceguard.d2.predictors import run_checks
from voiceguard.dsp.binarize import symbols_A
from voiceguard.dsp.pitch import jitter_sequence, periods, symbols_B
from voiceguard.types import Segment

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """Live speech reference model for D2 detector (FR-06, FR-07)."""

    channel: str
    variant: str
    frames: str
    d2_config: dict[str, Any]
    n_segments: int
    n_speakers: int
    underpowered: bool
    mu0: dict[str, float]
    sigma0: dict[str, float]
    median0: dict[str, float]
    mad0: dict[str, float]
    sigma_samp: dict[str, float]
    sigma_spk: dict[str, float]
    pi0: dict[str, list[list[float]]]
    code_version: str = "0.1.0"

    def save(self, path: str | Path) -> None:
        """Save reference model to JSON file."""
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, cfg: Config | None = None) -> Reference:
        """Load reference model from JSON file and validate D2 config compatibility."""
        if cfg is None:
            cfg = load_config()

        in_p = Path(path)
        if not in_p.is_file():
            raise FileNotFoundError(f"Reference file not found: {path}")

        with open(in_p, encoding="utf-8") as f:
            data = json.load(f)

        ref = cls(**data)

        # Validate that D2 configuration parameters match current config
        windows = cfg.d2.windows_B if ref.variant == "B" else cfg.d2.windows
        if ref.d2_config.get("windows") != windows:
            raise ValueError(
                f"Reference windows {ref.d2_config.get('windows')} do not match config {windows}"
            )
        if ref.d2_config.get("directions") != cfg.d2.directions:
            raise ValueError("Reference directions do not match config")
        if ref.d2_config.get("modes") != cfg.d2.modes:
            raise ValueError(
                f"Reference modes {ref.d2_config.get('modes')} do not match config {cfg.d2.modes}"
            )

        return ref


def build(
    channel: str,
    variant: str = "A",
    frames: str = "all",
    cfg: Config | None = None,
    segments: list[Segment] | None = None,
    speakers: list[str] | None = None,
) -> Reference:
    """Build live speech reference model for a given channel (FR-06).

    Args:
        channel: Channel specification string (e.g. 'clean', 'amrnb_12.2').
        variant: 'A' (LPC residual) or 'B' (pitch jitter).
        frames: 'all', 'voiced', or 'unvoiced' (for variant A).
        cfg: System configuration.
        segments: Calibration segments (only calib partition, live speech).
        speakers: Optional list of speaker IDs matching segments.

    Returns:
        Reference: Fitted reference model.
    """
    if cfg is None:
        cfg = load_config()

    if segments is None:
        segments = []

    n_segments = len(segments)
    speaker_set = set(speakers) if speakers else {s.clip_id for s in segments}
    n_speakers = len(speaker_set)

    underpowered = n_segments < 500 or n_speakers < 50
    if underpowered:
        logger.warning(
            f"Reference underpowered: {n_segments} segs (min 500), {n_speakers} spks (min 50)"
        )

    # Collect checks across segments
    all_deltas: dict[str, list[float]] = {}
    all_n_test: dict[str, list[int]] = {}
    pi0_counts: dict[int, np.ndarray] = {
        order: np.ones((1 << order, 2), dtype=np.float64) for order in cfg.d2.chi2_orders
    }

    for seg in segments:
        if variant == "A":
            bits = symbols_A(seg, cfg=cfg, frames=frames)  # type: ignore[arg-type]
            d_seq = None
        else:
            p_seq = periods(seg)
            d_seq = jitter_sequence(p_seq)
            bits = symbols_B(p_seq)

        if len(bits) == 0:
            continue

        checks = run_checks(bits, variant=variant, cfg=cfg, d_seq=d_seq)
        for cid, (d_val, n_test) in checks.items():
            if not np.isnan(d_val):
                all_deltas.setdefault(cid, []).append(d_val)
                all_n_test.setdefault(cid, []).append(n_test)

        # Accumulate transitions for chi2
        for order in cfg.d2.chi2_orders:
            if len(bits) > order:
                sub = np.lib.stride_tricks.sliding_window_view(bits, order + 1)
                powers = 1 << np.arange(order)[::-1]
                contexts = sub[:, :order].dot(powers)
                next_bits = sub[:, order]
                for c, b in zip(contexts, next_bits, strict=True):
                    pi0_counts[order][c, b] += 1.0

    mu0: dict[str, float] = {}
    sigma0: dict[str, float] = {}
    median0: dict[str, float] = {}
    mad0: dict[str, float] = {}
    sigma_samp: dict[str, float] = {}
    sigma_spk: dict[str, float] = {}

    for cid, vals in all_deltas.items():
        arr = np.array(vals, dtype=np.float64)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-4
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)) * 1.4826) or 1e-4

        mean_n = float(np.mean(all_n_test[cid])) if cid in all_n_test and all_n_test[cid] else 200.0
        s_samp = float(1.0 / (2.0 * np.sqrt(max(1.0, mean_n))))
        s_spk = float(np.sqrt(max(0.0, s * s - s_samp * s_samp)))

        mu0[cid] = m
        sigma0[cid] = s
        median0[cid] = med
        mad0[cid] = mad
        sigma_samp[cid] = s_samp
        sigma_spk[cid] = s_spk

    # Transition probabilities pi0
    pi0_dict: dict[str, list[list[float]]] = {}
    for order, counts in pi0_counts.items():
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / np.maximum(row_sums, 1e-9)
        pi0_dict[str(order)] = probs.tolist()

    windows = cfg.d2.windows_B if variant == "B" else cfg.d2.windows
    d2_config = {
        "windows": windows,
        "directions": cfg.d2.directions,
        "modes": cfg.d2.modes,
        "train_frac": cfg.d2.train_frac,
        "max_train": cfg.d2.max_train,
        "mlp_lr": cfg.d2.mlp_lr,
        "max_iter": cfg.d2.max_iter,
        "seed": cfg.d2.seed,
    }

    return Reference(
        channel=channel,
        variant=variant,
        frames=frames,
        d2_config=d2_config,
        n_segments=n_segments,
        n_speakers=n_speakers,
        underpowered=underpowered,
        mu0=mu0,
        sigma0=sigma0,
        median0=median0,
        mad0=mad0,
        sigma_samp=sigma_samp,
        sigma_spk=sigma_spk,
        pi0=pi0_dict,
        code_version="0.1.0",
    )
