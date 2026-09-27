"""D1: Acoustic anti-spoofing detector (упрощённое ЧТЗ-06).

FR-01: LFCC feature extraction.
FR-05: LFCC statistics + Logistic Regression classifier.
Training: segments from train split with channel augmentation (clean, g711a, amrnb_12.2).
Saves model to data/models/d1_nb.joblib.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from voiceguard.config import Config, load_config
from voiceguard.types import Segment

logger = logging.getLogger(__name__)

# Number of LFCC coefficients
N_LFCC = 20
# Number of filters
N_FILTERS = 40


# ---------------------------------------------------------------------------
# FR-01: LFCC feature extraction
# ---------------------------------------------------------------------------


def _lfcc_frame(frame: np.ndarray, sr: int, n_lfcc: int = N_LFCC, n_filters: int = N_FILTERS) -> np.ndarray:
    """Compute LFCC for a single frame using linear filter bank (not mel)."""
    n = len(frame)
    if n == 0:
        return np.zeros(n_lfcc)

    # FFT
    spectrum = np.abs(np.fft.rfft(frame * np.hamming(n))) ** 2

    # Linear filter bank
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    f_max = sr / 2.0
    boundaries = np.linspace(0, f_max, n_filters + 2)

    filters: list[np.ndarray] = []
    for i in range(n_filters):
        f_lo = boundaries[i]
        f_mid = boundaries[i + 1]
        f_hi = boundaries[i + 2]
        filt = np.zeros_like(freqs)
        # Rising slope
        mask_up = (freqs >= f_lo) & (freqs <= f_mid)
        if (f_mid - f_lo) > 0:
            filt[mask_up] = (freqs[mask_up] - f_lo) / (f_mid - f_lo)
        # Falling slope
        mask_dn = (freqs > f_mid) & (freqs <= f_hi)
        if (f_hi - f_mid) > 0:
            filt[mask_dn] = (f_hi - freqs[mask_dn]) / (f_hi - f_mid)
        filters.append(filt)

    bank = np.array(filters)  # (n_filters, n_fft)
    log_energy = np.log(bank @ spectrum + 1e-8)  # (n_filters,)

    # DCT-II
    from scipy.fft import dct

    lfcc = dct(log_energy, type=2, n=n_lfcc, norm="ortho")
    return lfcc


def extract_lfcc(
    seg: Segment,
    n_lfcc: int = N_LFCC,
    frame_ms: int = 20,
) -> np.ndarray:
    """Extract LFCC features from a segment, return (n_frames, n_lfcc) array."""
    sr = seg.sr
    frame_len = int(sr * frame_ms / 1000)
    samples = seg.samples

    frames: list[np.ndarray] = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        f = samples[start : start + frame_len]
        frames.append(_lfcc_frame(f, sr, n_lfcc=n_lfcc))

    if not frames:
        return np.zeros((1, n_lfcc))
    return np.array(frames)


# ---------------------------------------------------------------------------
# FR-05: Segment-level feature vector (statistics of LFCC)
# ---------------------------------------------------------------------------


def segment_features(seg: Segment, cfg: Config | None = None) -> np.ndarray:
    """Compute segment-level feature vector: mean + std of each LFCC coefficient.

    Returns 1-D vector of shape (2 * N_LFCC,).
    """
    if cfg is None:
        cfg = load_config()
    lfcc = extract_lfcc(seg, n_lfcc=N_LFCC, frame_ms=cfg.audio.frame_ms)
    mu = lfcc.mean(axis=0)
    std = lfcc.std(axis=0)
    return np.concatenate([mu, std])


# ---------------------------------------------------------------------------
# Channel augmentation helper
# ---------------------------------------------------------------------------


def _augment_segment(seg: Segment, channels: list[str]) -> list[tuple[Segment, str]]:
    """Apply channel simulation to a segment, return list of (augmented_seg, channel_name)."""
    from voiceguard.channel.simulator import apply
    from voiceguard.channel.spec import ChannelSpec

    result: list[tuple[Segment, str]] = []

    for ch_str in channels:
        if ch_str == "clean":
            result.append((seg, ch_str))
            continue
        try:
            spec = ChannelSpec.parse(ch_str)
            from voiceguard.types import AudioClip

            clip = AudioClip(samples=seg.samples, sr=seg.sr, clip_id=f"{seg.clip_id}_aug")
            aug_clip = apply(clip, spec)
            from voiceguard.dsp.vad import segment as seg_fn

            aug_segs = seg_fn(aug_clip)
            if aug_segs:
                result.append((aug_segs[0], ch_str))
            else:
                result.append((seg, ch_str))
        except Exception as exc:
            logger.debug("Channel augmentation failed for %s: %s", ch_str, exc)
            result.append((seg, ch_str))

    return result


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


def train(
    segments: list[Segment],
    labels: list[str],
    cfg: Config | None = None,
    aug_channels: list[str] | None = None,
    model_path: str | Path = "data/models/d1_nb.joblib",
) -> Any:
    """Train D1 logistic regression and save to model_path.

    Args:
        segments: Training segments.
        labels: "live" or "spoof" per segment.
        cfg: Config.
        aug_channels: Channels for augmentation (e.g. ["clean", "g711a", "amrnb_12.2"]).
        model_path: Output path for joblib model.

    Returns:
        Fitted sklearn Pipeline.
    """
    if cfg is None:
        cfg = load_config()
    if aug_channels is None:
        aug_channels = ["clean", "g711a", "amrnb_12.2"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for seg, lbl in zip(segments, labels, strict=False):
        y_val = 0 if lbl == "live" else 1
        aug_pairs = _augment_segment(seg, aug_channels)
        for aug_seg, _ in aug_pairs:
            feat = segment_features(aug_seg, cfg=cfg)
            X_list.append(feat)
            y_list.append(y_val)

    if not X_list:
        raise ValueError("No training samples collected")

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info("Training D1 on %d samples (%d unique segs)", len(y), len(segments))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=cfg.d2.seed)),
        ]
    )
    pipe.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    logger.info("D1 model saved to %s", model_path)

    return pipe


def load_model(path: str | Path = "data/models/d1_nb.joblib") -> Any:
    """Load a saved D1 model."""
    return joblib.load(path)


def predict_logit(seg: Segment, model: Any, cfg: Config | None = None) -> float:
    """Predict log-odds (logit) of spoof for a single segment.

    Returns:
        Float: positive = spoof, negative = live.
    """
    if cfg is None:
        cfg = load_config()
    feat = segment_features(seg, cfg=cfg)
    prob = model.predict_proba([feat])[0]  # [p_live, p_spoof]
    p_spoof = float(prob[1])
    p_spoof = np.clip(p_spoof, 1e-7, 1 - 1e-7)
    return float(np.log(p_spoof / (1 - p_spoof)))
