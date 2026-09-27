from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def generate_pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate deterministic pink noise (1/f spectral density) via FFT synthesis."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0

    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n_samples)
    std = float(np.std(pink))
    if std > 1e-9:
        pink = (pink - float(np.mean(pink))) / std
    return pink.astype(np.float32)


def calculate_active_speech_power(
    samples: np.ndarray,
    frame_len: int = 160,
    threshold_db: float = -30.0,
) -> float:
    """Calculate the average power of active speech frames."""
    if len(samples) == 0:
        return 0.0

    n_frames = max(1, len(samples) // frame_len)
    truncated_len = n_frames * frame_len
    frames = samples[:truncated_len].reshape(n_frames, frame_len)
    frame_powers = np.mean(frames.astype(np.float64) ** 2, axis=1)

    max_p = float(np.max(frame_powers))
    if max_p <= 1e-12:
        return float(np.mean(samples.astype(np.float64) ** 2))

    threshold = max_p * (10.0 ** (threshold_db / 10.0))
    active_mask = frame_powers >= threshold
    active_frames = frames[active_mask]
    if len(active_frames) == 0:
        return max_p

    return float(np.mean(active_frames.astype(np.float64) ** 2))


def _load_background_noise(
    n_samples: int,
    sr: int,
    rng: np.random.Generator,
    noise_dir: Path,
) -> np.ndarray | None:
    """Load a random slice of background noise from noise_dir, or return None."""
    if not noise_dir.is_dir():
        return None

    noise_files = sorted(
        list(noise_dir.glob("*.wav")) + list(noise_dir.glob("*.flac"))
    )
    if not noise_files:
        return None

    chosen_file = noise_files[int(rng.integers(0, len(noise_files)))]
    try:
        data, file_sr = sf.read(chosen_file, dtype="float32")
    except Exception:
        return None

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if file_sr != sr and len(data) > 0:
        from scipy import signal

        gcd = int(np.gcd(file_sr, sr))
        data = signal.resample_poly(data, sr // gcd, file_sr // gcd).astype(np.float32)

    if len(data) == 0:
        return None

    if len(data) < n_samples:
        repeats = int(np.ceil(n_samples / len(data)))
        data = np.tile(data, repeats)

    max_start = len(data) - n_samples
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    slice_data = data[start : start + n_samples].astype(np.float32)

    std = float(np.std(slice_data))
    if std > 1e-9:
        slice_data = (slice_data - float(np.mean(slice_data))) / std
    return np.asarray(slice_data, dtype=np.float32)


def add_noise(
    samples: np.ndarray,
    sr: int,
    snr_db: float,
    rng: np.random.Generator,
    noise_dir: str | Path | None = None,
) -> np.ndarray:
    """Add noise to samples matching target SNR on active speech."""
    if len(samples) == 0:
        return samples.copy()

    frame_len = max(1, int(sr * 0.02))
    p_speech = calculate_active_speech_power(samples, frame_len=frame_len)
    if p_speech <= 1e-12:
        p_speech = 1.0

    target_p_noise = p_speech / (10.0 ** (snr_db / 10.0))
    target_std_noise = np.sqrt(target_p_noise)

    noise: np.ndarray | None = None
    if noise_dir is not None:
        noise = _load_background_noise(len(samples), sr, rng, Path(noise_dir))

    if noise is None:
        noise = generate_pink_noise(len(samples), rng)

    noisy = samples.astype(np.float64) + noise.astype(np.float64) * target_std_noise
    return np.asarray(np.clip(noisy, -1.0, 1.0), dtype=np.float32)
