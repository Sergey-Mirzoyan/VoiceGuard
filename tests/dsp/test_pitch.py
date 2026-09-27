from __future__ import annotations

import numpy as np
from scipy import signal

from voiceguard.dsp.pitch import jitter_sequence, periods, symbols_B
from voiceguard.types import AudioClip


def _generate_synthetic_vowel(
    sr: int = 8000, f0: float = 125.0, duration_s: float = 2.0
) -> np.ndarray:
    """Generate synthetic vowel: 125 Hz pulse train through formant filter."""
    n_samples = int(sr * duration_s)

    # Impulse train at f0
    period_samples = int(round(sr / f0))
    pulse = np.zeros(n_samples, dtype=np.float32)
    pulse[::period_samples] = 1.0

    # Resonant formant filter at 500 Hz (bandwidth 80 Hz)
    b, a = signal.iirpeak(500.0, 500.0 / 80.0, fs=sr)
    vowel = signal.lfilter(b, a, pulse).astype(np.float32)

    # Normalize
    vowel = vowel / (np.max(np.abs(vowel)) + 1e-6) * 0.8
    return vowel


def test_ac05_synthetic_vowel_pitch_periods() -> None:
    """AC-05: Synthetic vowel (125 Hz pulse train) gives mean period = 64 +- 1."""
    sr = 8000
    vowel = _generate_synthetic_vowel(sr=sr, f0=125.0, duration_s=2.0)
    clip = AudioClip(samples=vowel, sr=sr, clip_id="vowel_125")

    t_periods = periods(clip)

    assert len(t_periods) > 10, f"Expected > 10 detected periods, got {len(t_periods)}"
    mean_p = float(np.mean(t_periods))
    # 8000 / 125 = 64.0 samples
    assert abs(mean_p - 64.0) <= 1.0, f"Expected mean period 64 +- 1, got {mean_p:.2f}"


def test_jitter_sequence_and_symbols_b() -> None:
    """Test jitter_sequence and symbols_B calculation."""
    # Synthetic periods with known slight variations
    T = np.array([64.0, 65.0, 63.0, 66.0, 64.0], dtype=np.float64)
    mean_t = np.mean(T)

    d = jitter_sequence(T)
    assert len(d) == len(T) - 1

    expected_d = np.diff(T) / mean_t
    np.testing.assert_allclose(d, expected_d, rtol=1e-5)

    c = symbols_B(T)
    assert len(c) == len(d)
    assert c.dtype == np.uint8
    expected_c = (expected_d > 0).astype(np.uint8)
    np.testing.assert_array_equal(c, expected_c)


def test_pitch_empty_and_short() -> None:
    """Test pitch functions with empty or degenerate input."""
    assert len(periods(np.zeros(0, dtype=np.float32), sr=8000)) == 0
    assert len(jitter_sequence(np.array([64.0]))) == 0
    assert len(symbols_B(np.array([64.0]))) == 0
