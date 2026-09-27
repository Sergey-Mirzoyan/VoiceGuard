from __future__ import annotations

import numpy as np

from voiceguard.channel.loss import apply_pcm_plc, generate_loss_mask
from voiceguard.channel.simulator import apply
from voiceguard.types import AudioClip, ChannelSpec


def test_loss_pct_range_and_reproducibility() -> None:
    """AC-04: +loss10 on 10s clip gives actual loss 7-13% at fixed seed, reproducible."""
    sr = 8000
    duration_s = 10.0
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    clip = AudioClip(samples=samples, sr=sr, clip_id="clip_10s")
    spec = ChannelSpec.parse("clean+loss10")

    seed = 42
    out1 = apply(clip, spec, seed=seed)
    actual_loss1 = out1.meta["actual_loss_pct"]

    # Check 7-13% range
    assert 7.0 <= actual_loss1 <= 13.0, f"Expected 7-13% loss, got {actual_loss1:.2f}%"

    # Reproducibility check: identical seed yields identical result
    out2 = apply(clip, spec, seed=seed)
    actual_loss2 = out2.meta["actual_loss_pct"]
    assert actual_loss1 == actual_loss2
    np.testing.assert_array_equal(out1.samples, out2.samples)

    # Different seed yields different pattern
    out3 = apply(clip, spec, seed=123)
    assert not np.array_equal(out1.samples, out3.samples)


def test_pcm_plc_attenuation() -> None:
    """Check that consecutive lost frames attenuate by 6 dB per frame."""
    frame_len = 160
    samples = np.ones(frame_len * 4, dtype=np.float32)

    # Frames 0 (good), 1 (lost), 2 (lost), 3 (good)
    loss_mask = np.array([False, True, True, False])
    plc_samples = apply_pcm_plc(samples, frame_len, loss_mask, attenuation_db=6.0)

    # Frame 0 is unchanged
    np.testing.assert_allclose(plc_samples[:frame_len], 1.0)

    # Frame 1 attenuated by ~6 dB (factor ~0.501)
    frame1 = plc_samples[frame_len : 2 * frame_len]
    expected_factor_1 = 10.0 ** (-6.0 / 20.0)
    # Check end of frame 1 (after boundary cross-fade)
    np.testing.assert_allclose(frame1[-16:], expected_factor_1, rtol=1e-3)

    # Frame 2 attenuated by ~12 dB (factor ~0.251)
    frame2 = plc_samples[2 * frame_len : 3 * frame_len]
    expected_factor_2 = 10.0 ** (-12.0 / 20.0)
    np.testing.assert_allclose(frame2, expected_factor_2, rtol=1e-3)


def test_gilbert_elliott_loss_mask() -> None:
    rng = np.random.default_rng(42)
    mask = generate_loss_mask(500, loss_pct=10.0, rng=rng, model="gilbert_elliott", r=0.3)
    loss_pct = np.mean(mask) * 100.0
    assert 5.0 <= loss_pct <= 20.0
