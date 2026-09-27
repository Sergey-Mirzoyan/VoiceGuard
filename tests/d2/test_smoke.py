"""Smoke tests for D2 module."""
from __future__ import annotations

import numpy as np
import pytest

from voiceguard.types import Segment


def _make_seg(sr: int = 16000, dur_s: float = 1.0, seed: int = 0) -> Segment:
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    samples = rng.standard_normal(n).astype(np.float32) * 0.1
    voiced_mask = np.ones(50, dtype=bool)
    return Segment(samples=samples, sr=sr, voiced_mask=voiced_mask, clip_id="test", index=0)


def test_build_reference() -> None:
    from voiceguard.d2.detector import build_reference

    segs = [_make_seg(seed=i) for i in range(4)]
    ref = build_reference(segs, channel="clean")
    assert ref.channel == "clean"
    assert len(ref.check_ids) > 0
    assert all(cid in ref.mu0 for cid in ref.check_ids)


def test_score_segment() -> None:
    from voiceguard.d2.detector import build_reference, score_segment

    segs = [_make_seg(seed=i) for i in range(4)]
    ref = build_reference(segs, channel="clean")
    result = score_segment(segs[0], ref)
    assert isinstance(result.s2, float)
    assert result.variant == "A"
    assert result.n_symbols > 0


def test_reference_serialize(tmp_path) -> None:
    from voiceguard.d2.detector import Reference, build_reference

    segs = [_make_seg(seed=i) for i in range(3)]
    ref = build_reference(segs, channel="clean")
    path = tmp_path / "ref.json"
    ref.save(path)
    ref2 = Reference.load(path)
    assert ref2.channel == ref.channel
    assert ref2.check_ids == ref.check_ids
