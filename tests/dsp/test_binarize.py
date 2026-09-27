from __future__ import annotations

import numpy as np
import pytest

from voiceguard.dsp.binarize import binarize, symbols_A
from voiceguard.dsp.lpc import residual
from voiceguard.types import Segment


def _generate_synthetic_speech(sr: int = 8000, duration_s: float = 3.0) -> np.ndarray:
    """Generate synthetic speech vowel."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    s = (
        0.5 * np.sin(2 * np.pi * 250 * t)
        + 0.3 * np.sin(2 * np.pi * 750 * t)
        + 0.2 * np.sin(2 * np.pi * 1250 * t)
        + 0.1 * np.random.randn(len(t)).astype(np.float32)
    )
    return (s / np.max(np.abs(s)) * 0.7).astype(np.float32)


def test_ac04_binarize_fraction_of_ones_on_speech() -> None:
    """AC-04: Fraction of ones after binarize on speech is 0.49..0.51."""
    sr = 8000
    speech = _generate_synthetic_speech(sr=sr, duration_s=4.0)
    res = residual(speech, sr=sr)

    # Frame length 20 ms -> 160 samples
    frame_len = int(sr * 0.02)
    bits = binarize(res, frame_len=frame_len)

    assert len(bits) == len(res)
    assert set(np.unique(bits)).issubset({0, 1})

    ratio = float(np.mean(bits))
    assert 0.49 <= ratio <= 0.51, f"Fraction of ones {ratio:.4f} is outside [0.49, 0.51]"


def test_binarize_empty_and_edge_cases() -> None:
    """Test binarize with empty input and odd/even lengths."""
    empty = binarize(np.zeros(0, dtype=np.float32), frame_len=160)
    assert len(empty) == 0

    # Constant frame: all values equal median -> all get 0 per spec
    const_frame = np.ones(160, dtype=np.float32) * 5.0
    bits = binarize(const_frame, frame_len=160)
    assert np.all(bits == 0)


def test_symbols_a_filtering() -> None:
    """Test symbols_A frame filtering for all, voiced, unvoiced."""
    sr = 8000
    frame_ms = 20
    frame_len = int(sr * frame_ms / 1000)  # 160
    n_frames = 50  # 1 second = 50 frames

    # Create dummy segment with 30 voiced and 20 unvoiced frames
    voiced_mask = np.zeros(n_frames, dtype=bool)
    voiced_mask[:30] = True

    samples = np.random.randn(n_frames * frame_len).astype(np.float32)
    seg = Segment(samples=samples, sr=sr, voiced_mask=voiced_mask, clip_id="c1", index=0)

    syms_all = symbols_A(seg, frames="all")
    assert len(syms_all) == n_frames * frame_len

    syms_voiced = symbols_A(seg, frames="voiced")
    assert len(syms_voiced) == 30 * frame_len

    syms_unvoiced = symbols_A(seg, frames="unvoiced")
    assert len(syms_unvoiced) == 20 * frame_len

    # Concatenation property
    assert len(syms_voiced) + len(syms_unvoiced) == len(syms_all)

    # Invalid frame class raises ValueError
    with pytest.raises(ValueError):
        symbols_A(seg, frames="invalid")  # type: ignore[arg-type]
