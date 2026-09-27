from __future__ import annotations

import numpy as np

from voiceguard.dsp.vad import compute_vad_mask, compute_voiced_mask, segment
from voiceguard.types import AudioClip


def _create_speech_with_pauses(
    sr: int = 8000,
    total_duration_s: float = 10.0,
) -> np.ndarray:
    """Create 10s audio with clear speech bursts and silence pauses."""
    n_total = int(sr * total_duration_s)
    t = np.linspace(0, total_duration_s, n_total, endpoint=False)
    samples = np.zeros(n_total, dtype=np.float32)

    # 3 speech intervals: [0.5, 3.0] (2.5s), [4.0, 6.2] (2.2s), [7.0, 9.6] (2.6s)
    # Total speech = 2.5 + 2.2 + 2.6 = 7.3s
    intervals = [(0.5, 3.0), (4.0, 6.2), (7.0, 9.6)]

    for start_s, end_s in intervals:
        idx_start = int(start_s * sr)
        idx_end = int(end_s * sr)
        sub_t = t[idx_start:idx_end]
        # Harmonic speech-like vowel with f0=150Hz
        burst = (
            0.6 * np.sin(2 * np.pi * 150 * sub_t)
            + 0.3 * np.sin(2 * np.pi * 450 * sub_t)
            + 0.2 * np.sin(2 * np.pi * 900 * sub_t)
        ).astype(np.float32)
        samples[idx_start:idx_end] = burst

    return samples


def test_ac06_segmentation_10s_speech_with_pauses() -> None:
    """AC-06: 10 s speech with pauses segments into exactly 1.0 s speech chunks."""
    sr = 8000
    samples = _create_speech_with_pauses(sr=sr, total_duration_s=10.0)
    clip = AudioClip(samples=samples, sr=sr, clip_id="clip_10s_pauses")

    segments = segment(clip)

    assert len(segments) > 0, "Expected at least one segment"

    expected_len = int(sr * 1.0)  # Exactly 1.0 s = 8000 samples
    expected_frames = 50  # 1.0 s / 20 ms = 50 frames

    for seg in segments:
        assert len(seg.samples) == expected_len, (
            f"Segment {seg.index} length {len(seg.samples)} != 1.0 s ({expected_len} samples)"
        )
        assert len(seg.voiced_mask) == expected_frames, (
            f"Segment {seg.index} voiced_mask length {len(seg.voiced_mask)} != {expected_frames}"
        )
        assert seg.sr == sr
        assert seg.clip_id == "clip_10s_pauses"


def test_vad_and_voiced_masks_synthetic() -> None:
    """Test VAD and voicing masks detection on tone vs silence."""
    sr = 8000
    # 0.5s silence + 0.5s harmonic speech-like signal
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (
        0.6 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 450 * t)
        + 0.2 * np.sin(2 * np.pi * 900 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    audio = np.concatenate([silence, tone])

    vad_m = compute_vad_mask(audio, sr=sr, frame_ms=20)
    voiced_m = compute_voiced_mask(audio, sr=sr, frame_ms=20, vad_mask=vad_m)

    # 50 frames total: first 25 silence, next 25 tone
    assert len(vad_m) == 50
    assert len(voiced_m) == 50
    # Silence part should be non-speech
    assert np.sum(vad_m[:20]) == 0
    # Tone part should have speech detected
    assert np.sum(vad_m[25:]) > 15
    # Voiced mask should be non-zero in tone part
    assert np.sum(voiced_m[25:]) > 10


def test_segment_all_silence() -> None:
    """Test segmentation on complete silence yields no segments."""
    sr = 8000
    clip = AudioClip(samples=np.zeros(sr * 3, dtype=np.float32), sr=sr, clip_id="silence")
    segs = segment(clip)
    assert len(segs) == 0
