from __future__ import annotations

import numpy as np
from scipy import signal

from voiceguard.dsp.vad import compute_vad_mask, compute_voiced_mask, segment
from voiceguard.types import AudioClip


def _create_speech_with_pauses(
    sr: int = 8000,
    total_duration_s: float = 10.0,
) -> np.ndarray:
    """Create 10s audio with clear speech bursts and silence pauses."""
    n_total = int(sr * total_duration_s)
    samples = np.zeros(n_total, dtype=np.float32)

    intervals = [(0.5, 3.0), (4.0, 6.5), (7.5, 9.8)]
    f0 = 130.0
    step = int(sr / f0)
    b, a = signal.iirpeak(500.0, 500.0 / 80.0, fs=sr)

    for start_s, end_s in intervals:
        idx0 = int(start_s * sr)
        idx1 = int(end_s * sr)
        length = idx1 - idx0
        pulse = np.zeros(length, dtype=np.float32)
        pulse[::step] = 1.0
        pulse += 0.05 * np.random.randn(length).astype(np.float32)
        vowel = signal.lfilter(b, a, pulse).astype(np.float32)
        vowel = vowel / (np.max(np.abs(vowel)) + 1e-6) * 0.8
        samples[idx0:idx1] = vowel

    return samples


def test_ac06_segmentation_10s_speech_with_pauses() -> None:
    """AC-06: 10 s speech with pauses segments into 1.0 s chunks with unpadded remainder >= 0.5s."""
    sr = 8000
    samples = _create_speech_with_pauses(sr=sr, total_duration_s=10.0)
    clip = AudioClip(samples=samples, sr=sr, clip_id="clip_10s_pauses")

    segments = segment(clip)

    assert len(segments) > 1, "Expected multiple segments"

    frame_len = int(sr * 0.02)  # 160 samples per 20 ms frame
    full_seg_len = int(sr * 1.0)  # 8000 samples

    for seg in segments[:-1]:
        assert len(seg.samples) == full_seg_len, f"Full segment {seg.index} length != 1.0 s"
        assert len(seg.voiced_mask) == 50
        assert seg.sr == sr

    last_seg = segments[-1]
    # Remainder >= 0.5 s must not be zero-padded to 1.0 s
    assert len(last_seg.samples) >= int(sr * 0.5)
    assert len(last_seg.samples) == len(last_seg.voiced_mask) * frame_len
    if len(last_seg.voiced_mask) < 50:
        assert len(last_seg.samples) < full_seg_len, "Remainder >= 0.5s must NOT be padded to 1.0s"


def test_remainder_handling() -> None:
    """Test remainder < 0.5 s is discarded and remainder >= 0.5 s is kept unpadded."""
    sr = 8000
    frame_len = int(sr * 0.02)
    step = int(sr / 130.0)
    b, a = signal.iirpeak(500.0, 500.0 / 80.0, fs=sr)

    def make_speech(n_frames: int) -> np.ndarray:
        length = n_frames * frame_len
        pulse = np.zeros(length, dtype=np.float32)
        pulse[::step] = 1.0
        vowel = signal.lfilter(b, a, pulse).astype(np.float32)
        return (vowel / (np.max(np.abs(vowel)) + 1e-6) * 0.8).astype(np.float32)

    # 15 frames of speech (< 25 frames / 0.5s) -> discarded
    clip_short = AudioClip(samples=make_speech(15), sr=sr, clip_id="short")
    assert len(segment(clip_short)) == 0

    # 35 frames of speech (>= 25 frames / 0.5s and < 50 frames / 1.0s) -> 1 unpadded segment
    clip_rem = AudioClip(samples=make_speech(35), sr=sr, clip_id="rem")
    segs = segment(clip_rem)
    assert len(segs) == 1
    assert len(segs[0].voiced_mask) == 35
    assert len(segs[0].samples) == 35 * frame_len


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

    assert len(vad_m) == 50
    assert len(voiced_m) == 50
    assert np.sum(vad_m[:20]) == 0
    assert np.sum(vad_m[25:]) > 15
    assert np.sum(voiced_m[25:]) > 10


def test_segment_all_silence() -> None:
    """Test segmentation on complete silence yields no segments."""
    sr = 8000
    clip = AudioClip(samples=np.zeros(sr * 3, dtype=np.float32), sr=sr, clip_id="silence")
    segs = segment(clip)
    assert len(segs) == 0
