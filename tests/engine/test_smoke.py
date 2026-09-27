"""Smoke tests for engine: StreamEngine initialization and process_audio."""
from __future__ import annotations

import numpy as np


def test_stream_engine_init() -> None:
    """Engine should initialize without models (graceful degradation)."""
    from voiceguard.engine.stream_engine import StreamEngine

    engine = StreamEngine(models_dir="/nonexistent")
    assert engine is not None
    assert engine._sprt is not None


def test_process_audio_no_models() -> None:
    """process_audio with no models should return empty list or few WindowScores."""
    from voiceguard.engine.stream_engine import StreamEngine

    engine = StreamEngine(models_dir="/nonexistent")
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
    results = engine.process_audio(samples, sr=16000)
    # Should return list (possibly empty due to VAD)
    assert isinstance(results, list)
