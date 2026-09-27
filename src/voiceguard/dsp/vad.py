from __future__ import annotations

import sys
import types

import numpy as np
import parselmouth

from voiceguard.config import Config, load_config
from voiceguard.types import AudioClip, Segment

# Compatibility shim for webrtcvad under modern setuptools (>=70)
if "pkg_resources" not in sys.modules:
    dummy_pkg_resources = types.ModuleType("pkg_resources")
    dummy_pkg_resources.get_distribution = lambda name: types.SimpleNamespace(version="2.0.10")  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = dummy_pkg_resources

import webrtcvad  # noqa: E402


def compute_vad_mask(
    samples: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    aggressiveness: int = 2,
) -> np.ndarray:
    """Compute boolean speech activity mask for each frame_ms frame using WebRTC VAD."""
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad supports sr in (8000, 16000, 32000, 48000), got {sr}")

    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)

    vad = webrtcvad.Vad(int(aggressiveness))
    pcm16 = (np.clip(samples[: n_frames * frame_len], -1.0, 1.0) * 32767.0).astype(np.int16)

    mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        chunk = pcm16[i * frame_len : (i + 1) * frame_len].tobytes()
        try:
            mask[i] = vad.is_speech(chunk, sr)
        except Exception:
            mask[i] = False

    return mask


def compute_voiced_mask(
    samples: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    vad_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute boolean voicing mask for each frame (voiced if pitch != 0 and speech is active)."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)

    if vad_mask is None:
        vad_mask = np.ones(n_frames, dtype=bool)

    sound = parselmouth.Sound(
        samples[: n_frames * frame_len].astype(np.float64), sampling_frequency=float(sr)
    )
    time_step = frame_ms / 2000.0  # sample at twice frame rate
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=75.0, pitch_ceiling=600.0)

    voiced = np.zeros(n_frames, dtype=bool)
    frame_s = frame_ms / 1000.0

    for i in range(n_frames):
        if not vad_mask[i]:
            continue
        mid_time = (i + 0.5) * frame_s
        try:
            p_val = pitch.get_value_at_time(mid_time)
            if not np.isnan(p_val) and p_val > 0.0:
                voiced[i] = True
        except Exception:
            voiced[i] = False

    return voiced


def segment(
    clip: AudioClip,
    cfg: Config | None = None,
) -> list[Segment]:
    """Segment an audio clip into 1.0 s segments of concatenated speech frames with VAD voicing.
    """
    if cfg is None:
        cfg = load_config()

    sr = clip.sr
    frame_ms = cfg.audio.frame_ms
    frame_len = int(sr * frame_ms / 1000)
    segment_s = cfg.audio.segment_s
    frames_per_seg = int(segment_s / (frame_ms / 1000.0))  # 50 frames for 1.0 s

    samples = clip.samples.astype(np.float32)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return []

    vad_mask = compute_vad_mask(
        samples,
        sr=sr,
        frame_ms=frame_ms,
        aggressiveness=cfg.vad.aggressiveness,
    )
    voiced_mask = compute_voiced_mask(
        samples,
        sr=sr,
        frame_ms=frame_ms,
        vad_mask=vad_mask,
    )

    speech_frame_indices = np.where(vad_mask)[0]
    total_speech_frames = len(speech_frame_indices)
    if total_speech_frames == 0:
        return []

    segments: list[Segment] = []
    seg_idx = 0

    for start_f in range(0, total_speech_frames, frames_per_seg):
        end_f = start_f + frames_per_seg
        cur_frame_indices = speech_frame_indices[start_f:end_f]
        cur_count = len(cur_frame_indices)

        # Discard leftovers shorter than 0.5 s (less than half of frames_per_seg)
        if cur_count < frames_per_seg // 2:
            continue

        # Extract frames
        frame_blocks = [
            samples[idx * frame_len : (idx + 1) * frame_len] for idx in cur_frame_indices
        ]
        cur_voiced = voiced_mask[cur_frame_indices]

        # Remainder >= 0.5 s is kept as shorter segment without zero padding

        seg_samples = np.concatenate(frame_blocks).astype(np.float32)
        segments.append(
            Segment(
                samples=seg_samples,
                sr=sr,
                voiced_mask=cur_voiced,
                clip_id=clip.clip_id,
                index=seg_idx,
            )
        )
        seg_idx += 1

    return segments
