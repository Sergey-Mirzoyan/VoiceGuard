from __future__ import annotations

import numpy as np
from scipy import signal

from voiceguard.channel.simulator import apply
from voiceguard.channel.transcoder import StreamTranscoder
from voiceguard.types import AudioClip, ChannelSpec


def test_stream_transcoder_correlation_amrnb() -> None:
    """AC-06: StreamTranscoder on 5s clip matches batch apply with correlation >= 0.95."""
    sr = 8000
    duration_s = 5.0
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)

    # Multi-frequency test speech-like signal
    x = (
        0.5 * np.sin(2 * np.pi * 300.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1000.0 * t)
        + 0.2 * np.sin(2 * np.pi * 2500.0 * t)
    ).astype(np.float32)

    clip = AudioClip(samples=x, sr=sr, clip_id="clip_5s")
    spec = ChannelSpec.parse("amrnb_12.2")

    # 1. Batch apply
    batch_clip = apply(clip, spec, seed=42)
    batch_samples = batch_clip.samples

    # 2. Stream transcoder
    frame_len = 160  # 20 ms at 8 kHz
    n_frames = len(x) // frame_len
    streamed_frames = []

    with StreamTranscoder(spec, seed=42) as transcoder:
        for i in range(n_frames):
            frame = x[i * frame_len : (i + 1) * frame_len]
            out_frame = transcoder.push(frame)
            assert len(out_frame) == frame_len
            streamed_frames.append(out_frame)

    stream_samples = np.concatenate(streamed_frames)

    # 3. Cross-correlation alignment
    min_len = min(len(stream_samples), len(batch_samples))
    corr = signal.correlate(
        stream_samples[:min_len],
        batch_samples[:min_len],
        mode="full",
    )
    lag = int(np.argmax(corr)) - (min_len - 1)

    if lag >= 0:
        s1 = stream_samples[lag:min_len]
        s2 = batch_samples[: min_len - lag]
    else:
        s1 = stream_samples[: min_len + lag]
        s2 = batch_samples[-lag:min_len]

    eval_len = min(len(s1), len(s2), 35000)
    assert eval_len > 10000

    corr_coeff = float(np.corrcoef(s1[:eval_len], s2[:eval_len])[0, 1])
    assert corr_coeff >= 0.95, f"Expected correlation >= 0.95, got {corr_coeff:.4f}"


def test_stream_transcoder_g711a() -> None:
    frame_len = 160
    t = np.linspace(0, 0.02, frame_len, endpoint=False, dtype=np.float32)
    frame = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    spec = ChannelSpec.parse("g711a")
    with StreamTranscoder(spec) as transcoder:
        out_frame = transcoder.push(frame)
        assert len(out_frame) == frame_len
        # Check SNR of single frame
        snr = 10.0 * np.log10(np.sum(frame**2) / np.sum((frame - out_frame) ** 2))
        assert snr >= 30.0
