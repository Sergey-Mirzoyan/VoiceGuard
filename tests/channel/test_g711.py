from __future__ import annotations

import numpy as np

from voiceguard.channel.g711 import transcode_g711a, transcode_g711u


def _generate_synthetic_speech(sr: int = 8000, duration_s: float = 2.0) -> np.ndarray:
    """Generate harmonic speech-like vowel signal."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # Formants at 300, 1200, 2400 Hz
    s = (
        0.5 * np.sin(2 * np.pi * 300 * t)
        + 0.3 * np.sin(2 * np.pi * 1200 * t)
        + 0.2 * np.sin(2 * np.pi * 2400 * t)
    )
    # Normalize to -3 dBFS
    return (s / np.max(np.abs(s)) * 0.7).astype(np.float32)


def test_g711a_snr_on_speech() -> None:
    """AC-03: g711a on speech gives SNR relative to input >= 30 dB."""
    speech = _generate_synthetic_speech(sr=8000, duration_s=2.0)
    transcoded = transcode_g711a(speech)

    assert len(transcoded) == len(speech)
    noise = speech - transcoded
    p_signal = float(np.sum(speech**2))
    p_noise = float(np.sum(noise**2))

    snr_db = 10.0 * np.log10(p_signal / p_noise)
    assert snr_db >= 30.0, f"Expected SNR >= 30 dB, got {snr_db:.2f} dB"


def test_g711u_snr_on_speech() -> None:
    speech = _generate_synthetic_speech(sr=8000, duration_s=2.0)
    transcoded = transcode_g711u(speech)

    assert len(transcoded) == len(speech)
    noise = speech - transcoded
    p_signal = float(np.sum(speech**2))
    p_noise = float(np.sum(noise**2))

    snr_db = 10.0 * np.log10(p_signal / p_noise)
    assert snr_db >= 30.0, f"Expected SNR >= 30 dB, got {snr_db:.2f} dB"
