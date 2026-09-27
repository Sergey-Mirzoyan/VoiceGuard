from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from voiceguard.config import load_config
from voiceguard.d2.__main__ import analyze_main, build_ref_main, prng_check_main
from voiceguard.d2.analysis import analyze, threshold
from voiceguard.d2.reference import Reference, build
from voiceguard.d2.regression import rho
from voiceguard.types import Segment


def _make_dummy_segments(n_segs: int = 4, sr: int = 8000) -> list[Segment]:
    segments = []
    t = np.linspace(0, 1.0, sr, endpoint=False)
    v_mask = np.ones(50, dtype=bool)

    for i in range(n_segs):
        freq = 150.0 + i * 25.0
        samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        segments.append(
            Segment(
                samples=samples,
                sr=sr,
                voiced_mask=v_mask,
                clip_id=f"spk_{i % 2}_clip_{i}",
                index=i,
            )
        )
    return segments


def test_ac05_build_ref_and_load_mismatch(tmp_path: Path) -> None:
    """AC-05: build-ref creates reference file; Reference.load rejects modified windows."""
    cfg = load_config()
    segs = _make_dummy_segments(n_segs=4)

    ref = build(channel="clean", variant="A", frames="all", cfg=cfg, segments=segs)
    ref_path = tmp_path / "test_reference.json"
    ref.save(ref_path)

    assert ref_path.is_file()

    # Successful load with matching config
    loaded = Reference.load(ref_path, cfg=cfg)
    assert loaded.channel == "clean"
    assert loaded.variant == "A"
    assert loaded.n_segments == 4

    # Load with modified windows raises ValueError
    cfg_modified = load_config(overrides={"d2": {"windows": [4, 8, 16]}})
    with pytest.raises(ValueError, match="Reference windows"):
        Reference.load(ref_path, cfg=cfg_modified)


def test_ac06_analyze_populates_all_fields_and_finite_s2(tmp_path: Path) -> None:
    """AC-06: analyze on segment produces finite s2 and populates all D2Result fields."""
    cfg = load_config()
    segs = _make_dummy_segments(n_segs=4)
    ref = build(channel="clean", variant="A", frames="all", cfg=cfg, segments=segs)

    result = analyze(segs[0], ref=ref, cfg=cfg)

    assert np.isfinite(result.s2)
    assert result.variant == "A"
    assert result.n_symbols > 0
    assert len(result.delta) == 12
    assert len(result.z) == 12
    assert "order_2" in result.chi2_p
    assert "order_4" in result.chi2_p


def test_ac07_rho_regression() -> None:
    """AC-07: rho on AR(1) with coeff 0.8 is in [0.55, 0.72]; on white noise |rho| < 0.05."""
    cfg = load_config()
    np.random.seed(42)

    white_noise = np.random.randn(4000)
    ar1 = signal.lfilter([1.0], [1.0, -0.8], white_noise)

    val_ar1 = rho(ar1, w=4, cfg=cfg)
    assert 0.55 <= val_ar1 <= 0.72, f"Expected rho in [0.55, 0.72], got {val_ar1:.4f}"

    val_white = rho(white_noise, w=4, cfg=cfg)
    assert abs(val_white) < 0.05, f"Expected |rho| < 0.05, got {val_white:.4f}"


def test_d2_threshold() -> None:
    """Test threshold calculation matches theoretical quantile ~3.34."""
    cfg = load_config()
    th = threshold(cfg, m=12)
    assert 3.30 <= th <= 3.40, f"Expected threshold ~3.34, got {th:.4f}"


def test_cli_tools(tmp_path: Path) -> None:
    """Test CLI tools for prng-check, build-ref, and analyze."""
    # 1. prng-check
    exit_code = prng_check_main(["--gen", "urandom", "--n", "5000"])
    assert exit_code == 0

    # 2. build-ref
    out_ref = tmp_path / "cli_ref.json"
    exit_code = build_ref_main(["--channel", "clean", "--out", str(out_ref)])
    assert exit_code == 0
    assert out_ref.is_file()

    # 3. analyze
    exit_code = analyze_main(["--clip-id", "cli_test", "--channel", "clean"])
    assert exit_code == 0
