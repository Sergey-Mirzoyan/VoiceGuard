from __future__ import annotations

import numpy as np
import parselmouth

from voiceguard.types import AudioClip, Segment


def periods(
    clip_or_segment: AudioClip | Segment | np.ndarray,
    sr: int | None = None,
) -> np.ndarray:
    """Extract fundamental pitch periods T_i (samples) on voiced regions using Parselmouth."""
    if isinstance(clip_or_segment, (AudioClip, Segment)):
        samples = clip_or_segment.samples
        sample_rate = clip_or_segment.sr
    else:
        samples = np.asarray(clip_or_segment, dtype=np.float32)
        if sr is None:
            raise ValueError("Sample rate sr must be provided when passing a raw numpy array")
        sample_rate = sr

    if len(samples) == 0:
        return np.zeros(0, dtype=np.float64)

    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(sample_rate))

    # Praat periodic cross-correlation pitch marks: min pitch 75 Hz, max pitch 600 Hz
    min_pitch = 75.0
    max_pitch = 600.0
    max_period_s = 1.0 / min_pitch
    min_period_s = 1.0 / max_pitch

    try:
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            min_pitch,
            max_pitch,
        )
        n_points = int(parselmouth.praat.call(point_process, "Get number of points"))
    except Exception:
        return np.zeros(0, dtype=np.float64)

    if n_points < 2:
        return np.zeros(0, dtype=np.float64)

    times = np.array(
        [
            float(parselmouth.praat.call(point_process, "Get time from index", i))
            for i in range(1, n_points + 1)
        ],
        dtype=np.float64,
    )

    diffs_s = np.diff(times)
    # Exclude intervals crossing unvoiced gaps (greater than max pitch period)
    valid_mask = (diffs_s >= min_period_s * 0.8) & (diffs_s <= max_period_s * 1.2)
    valid_diffs = diffs_s[valid_mask]

    return valid_diffs * float(sample_rate)


def jitter_sequence(T: np.ndarray) -> np.ndarray:
    """Compute relative pitch period jitter sequence d_i = (T_{i+1} - T_i) / mean(T)."""
    t_arr = np.asarray(T, dtype=np.float64)
    if len(t_arr) < 2:
        return np.zeros(0, dtype=np.float64)

    mean_t = float(np.mean(t_arr))
    if mean_t <= 1e-12:
        return np.zeros(len(t_arr) - 1, dtype=np.float64)

    diffs = np.diff(t_arr)
    return diffs / mean_t


def symbols_B(T: np.ndarray) -> np.ndarray:
    """Extract Variant B binary symbols: c_i = 1{d_i > 0} from period sequence T."""
    d = jitter_sequence(T)
    if len(d) == 0:
        return np.zeros(0, dtype=np.uint8)
    return (d > 0.0).astype(np.uint8)
