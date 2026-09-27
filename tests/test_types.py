from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from voiceguard.types import (
    AudioClip,
    ChannelSpec,
    D2Result,
    Label,
    Segment,
    WindowScore,
)


def test_label_enum() -> None:
    assert Label.LIVE == "live"
    assert Label.SPOOF == "spoof"
    assert Label.LIVE.value == "live"
    assert Label.SPOOF.value == "spoof"


def test_audio_clip() -> None:
    samples = np.zeros(8000, dtype=np.float32)
    clip = AudioClip(
        samples=samples,
        sr=8000,
        clip_id="test_clip_1",
        meta={"speaker": "arm_001"},
    )
    assert clip.sr == 8000
    assert clip.clip_id == "test_clip_1"
    assert clip.meta["speaker"] == "arm_001"
    assert len(clip.samples) == 8000

    # Ensure immutability (frozen dataclass)
    with pytest.raises(FrozenInstanceError):
        clip.sr = 16000  # type: ignore[misc]


def test_channel_spec() -> None:
    spec = ChannelSpec(
        codec="amrnb",
        bitrate_kbps=12.2,
        loss_pct=3.0,
        snr_db=20.0,
    )
    assert spec.codec == "amrnb"
    assert spec.bitrate_kbps == 12.2
    assert spec.loss_pct == 3.0
    assert spec.snr_db == 20.0

    # Default values test
    clean = ChannelSpec(codec="clean", bitrate_kbps=None)
    assert clean.loss_pct == 0.0
    assert clean.snr_db is None


def test_segment() -> None:
    samples = np.zeros(8000, dtype=np.float32)
    voiced = np.ones(50, dtype=bool)
    seg = Segment(
        samples=samples,
        sr=8000,
        voiced_mask=voiced,
        clip_id="test_clip_1",
        index=0,
    )
    assert seg.sr == 8000
    assert seg.clip_id == "test_clip_1"
    assert seg.index == 0
    assert len(seg.samples) == 8000
    assert len(seg.voiced_mask) == 50


def test_d2_result() -> None:
    res = D2Result(
        delta={"A_bit_w16_fwd": 0.05},
        z={"A_bit_w16_fwd": 2.1},
        s2=2.1,
        chi2_p={"A_bit_w16_fwd": 0.03},
        n_symbols=500,
        variant="A",
    )
    assert res.variant == "A"
    assert res.s2 == 2.1
    assert res.n_symbols == 500
    assert res.delta["A_bit_w16_fwd"] == 0.05


def test_window_score() -> None:
    score = WindowScore(
        t=1.0,
        s1=-0.5,
        s2=2.1,
        llr=1.2,
        Lambda=1.2,
        verdict="pending",
    )
    assert score.t == 1.0
    assert score.s1 == -0.5
    assert score.s2 == 2.1
    assert score.llr == 1.2
    assert score.Lambda == 1.2
    assert score.verdict == "pending"
