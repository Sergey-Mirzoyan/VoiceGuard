"""Smoke tests for D1 module."""
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


def test_extract_lfcc_shape() -> None:
    from voiceguard.d1.detector import extract_lfcc

    seg = _make_seg()
    lfcc = extract_lfcc(seg)
    assert lfcc.ndim == 2
    assert lfcc.shape[1] == 20  # N_LFCC


def test_segment_features_shape() -> None:
    from voiceguard.d1.detector import segment_features

    seg = _make_seg()
    feat = segment_features(seg)
    assert feat.shape == (40,)  # 2 * N_LFCC


def test_train_and_predict() -> None:
    from voiceguard.d1.detector import predict_logit, train

    segs = [_make_seg(seed=i) for i in range(6)]
    labels = ["live", "live", "live", "spoof", "spoof", "spoof"]
    model = train(segs, labels, aug_channels=["clean"], model_path="/tmp/d1_test.joblib")

    logit = predict_logit(segs[0], model)
    assert isinstance(logit, float)
