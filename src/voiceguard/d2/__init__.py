from __future__ import annotations

from voiceguard.d2.analysis import analyze, threshold
from voiceguard.d2.chi2 import markov_test
from voiceguard.d2.detector import Reference, build_reference, score_segment
from voiceguard.d2.predictors import (
    bit_delta,
    block_delta,
    dissertation_verdict,
    run_checks,
)
from voiceguard.d2.reference import build
from voiceguard.d2.regression import rho

__all__ = [
    "Reference",
    "analyze",
    "bit_delta",
    "block_delta",
    "build",
    "build_reference",
    "dissertation_verdict",
    "markov_test",
    "rho",
    "run_checks",
    "score_segment",
    "threshold",
]
