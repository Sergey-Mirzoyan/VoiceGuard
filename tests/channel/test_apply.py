from __future__ import annotations

import numpy as np
from scipy import signal

from voiceguard.channel.simulator import apply
from voiceguard.types import AudioClip, ChannelSpec


def test_amrnb_sine_pitch_and_duration() -> None:
    """AC-02: Sine 440 Hz through amrnb_12.2 preserves frequency (±5 Hz) and duration (±20 ms)."""
    sr = 8000
    duration_s = 2.0
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    sine = (0.7 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    clip = AudioClip(samples=sine, sr=sr, clip_id="sine_440")
    spec = ChannelSpec.parse("amrnb_12.2")

    out_clip = apply(clip, spec, seed=42)

    # 1. Output duration matches input within 20 ms (160 samples at 8 kHz)
    duration_diff_ms = abs(len(out_clip.samples) - len(sine)) / sr * 1000.0
    assert duration_diff_ms <= 20.0, f"Duration diff {duration_diff_ms} ms exceeds 20 ms"

    # 2. Fundamental frequency preservation (±5 Hz)
    # Estimate frequency via peak of FFT
    fft_mag = np.abs(np.fft.rfft(out_clip.samples))
    freqs = np.fft.rfftfreq(len(out_clip.samples), 1.0 / sr)
    peak_idx = int(np.argmax(fft_mag))
    peak_freq = freqs[peak_idx]

    assert abs(peak_freq - 440.0) <= 5.0, f"Peak freq {peak_freq:.1f} Hz outside 440±5 Hz"


def test_amrnb_spectrum_empty_above_4khz() -> None:
    """AC-05: After amrnb_*, energy above 3.8 kHz is >= 40 dB below total energy."""
    sr = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)

    # Speech-like harmonic vowel
    f0 = 140.0
    pulses = np.zeros_like(t)
    pulses[:: int(sr / f0)] = 1.0

    b1, a1 = signal.iirpeak(500.0, 500.0 / 80.0, fs=sr)
    b2, a2 = signal.iirpeak(1500.0, 1500.0 / 120.0, fs=sr)
    b3, a3 = signal.iirpeak(2500.0, 2500.0 / 150.0, fs=sr)

    vowel = (
        signal.lfilter(b1, a1, pulses)
        + signal.lfilter(b2, a2, pulses)
        + 0.5 * signal.lfilter(b3, a3, pulses)
    )
    vowel = (vowel / np.max(np.abs(vowel)) * 0.8).astype(np.float32)

    clip = AudioClip(samples=vowel, sr=sr, clip_id="vowel_16k")
    spec = ChannelSpec.parse("amrnb_12.2")

    out_clip = apply(clip, spec, seed=42)
    assert out_clip.sr == 8000

    # Calculate power spectral density
    freqs, psd = signal.welch(out_clip.samples, fs=out_clip.sr, nperseg=1024)
    total_energy = float(np.sum(psd))
    high_energy = float(np.sum(psd[freqs >= 3800.0]))

    ratio_db = 10.0 * np.log10(high_energy / total_energy)
    assert ratio_db <= -40.0, f"Energy above 3.8 kHz was {ratio_db:.1f} dB (expected <= -40 dB)"


def test_amrwb_transcoding() -> None:
    sr = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sine = (0.7 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    clip = AudioClip(samples=sine, sr=sr, clip_id="wb_sine")
    spec = ChannelSpec.parse("amrwb_12.65")

    out_clip = apply(clip, spec, seed=42)
    assert out_clip.sr == 16000

    fft_mag = np.abs(np.fft.rfft(out_clip.samples))
    freqs = np.fft.rfftfreq(len(out_clip.samples), 1.0 / sr)
    peak_freq = freqs[int(np.argmax(fft_mag))]
    assert abs(peak_freq - 440.0) <= 5.0


def test_clean_channel_preserves_audio() -> None:
    sr = 16000
    samples = np.random.uniform(-0.5, 0.5, 8000).astype(np.float32)
    clip = AudioClip(samples=samples, sr=sr, clip_id="clean_test")
    spec = ChannelSpec.parse("clean")

    out_clip = apply(clip, spec, seed=42)
    assert out_clip.sr == sr
    np.testing.assert_array_equal(out_clip.samples, clip.samples)
    assert out_clip.meta["channel"] == "clean"
