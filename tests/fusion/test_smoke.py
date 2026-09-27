"""Smoke tests for fusion module: LLRCalibrator and SPRT."""
from __future__ import annotations

import math

from voiceguard.fusion.calibrator import SPRT, LLRCalibrator


def test_sprt_thresholds() -> None:
    alpha, beta = 0.001, 0.05
    sprt = SPRT(alpha=alpha, beta=beta, t_max_s=30.0)
    assert math.isclose(sprt.A, math.log((1 - beta) / alpha), rel_tol=1e-6)
    assert math.isclose(sprt.B, math.log(beta / (1 - alpha)), rel_tol=1e-6)


def test_sprt_suspicious() -> None:
    sprt = SPRT(alpha=0.001, beta=0.05, t_max_s=30.0)
    # Keep pushing high LLR until suspicious
    for _ in range(100):
        v = sprt.update(1.0)
        if v == "suspicious":
            break
    assert sprt.verdict == "suspicious"


def test_sprt_normal() -> None:
    sprt = SPRT(alpha=0.001, beta=0.05, t_max_s=30.0)
    for _ in range(100):
        v = sprt.update(-1.0)
        if v == "normal":
            break
    assert sprt.verdict == "normal"


def test_sprt_undecided() -> None:
    sprt = SPRT(alpha=0.001, beta=0.05, t_max_s=5.0)
    for _ in range(10):
        sprt.update(0.0)
    assert sprt.verdict == "undecided"


def test_sprt_reset() -> None:
    sprt = SPRT()
    for _ in range(100):
        sprt.update(1.0)
    assert sprt.verdict == "suspicious"
    sprt.reset()
    assert sprt.verdict == "pending"
    assert sprt.Lambda == 0.0


def test_calibrator_trivial() -> None:
    cal = LLRCalibrator()
    # Trivial (unfitted) calibrator
    llr = cal.llr(1.0, 2.0)
    assert isinstance(llr, float)


def test_calibrator_fit() -> None:
    cal = LLRCalibrator()
    s1 = [1.0, 2.0, -1.0, -2.0]
    s2 = [3.0, 3.5, 0.1, 0.2]
    y  = [1, 1, 0, 0]
    cal.fit(s1, s2, y)
    llr_spoof = cal.llr(2.0, 3.5)
    llr_live  = cal.llr(-2.0, 0.1)
    assert llr_spoof > llr_live

    cal1 = LLRCalibrator()
    cal1.fit(s1, labels=y)
    assert cal1.llr(2.0) > cal1.llr(-2.0)
