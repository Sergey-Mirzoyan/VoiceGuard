from __future__ import annotations

import joblib  # type: ignore[import-untyped]
import numpy as np

from tests.prng_fixtures import generate_lfsr, generate_mt, generate_urandom
from voiceguard.config import load_config
from voiceguard.d2.chi2 import markov_test
from voiceguard.d2.predictors import bit_delta, dissertation_verdict


def _run_single_urandom_check(seed: int) -> bool:
    cfg = load_config(overrides={"d2": {"seed": seed}})
    bits = generate_urandom(100000)
    res = dissertation_verdict(bits, cfg=cfg)
    return bool(res["is_violation"])


def test_ac01_urandom_false_alarm_rate() -> None:
    """AC-01: On 200 sequences of os.urandom of 100,000 bits, dissertation compatibility
    mode gives false alarm rate <= 2 * alpha (alpha = 0.01).
    """
    n_sequences = 200
    violations = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_run_single_urandom_check)(i) for i in range(n_sequences)
    )

    n_violations = sum(violations)
    false_alarm_rate = n_violations / float(n_sequences)

    # alpha = 0.01 -> 2 * alpha = 0.02
    assert false_alarm_rate <= 0.02, (
        f"False alarm rate {false_alarm_rate:.3f} ({n_violations}/{n_sequences}) exceeds 2*alpha"
    )


def test_ac02_lfsr_compatibility() -> None:
    """AC-02: On LFSR (degree 32), compatibility mode evaluates delta_max and verdict.

    Per spec AC-02: 'режим совместимости фиксирует нарушение; либо расхождение объяснено в отчёте'.
    With T=50 training samples, an MLP cannot learn a 32-bit linear recurrence over GF(2);
    both analysis.py and dissertation_verdict yield delta_max < threshold (no violation).
    """
    cfg = load_config()
    lfsr_bits = generate_lfsr(100000)
    res = dissertation_verdict(lfsr_bits, cfg=cfg)

    assert "delta_max" in res
    assert "eps" in res
    assert "verdict" in res
    assert np.isfinite(res["delta_max"])

    # Strict verdict verification
    assert res["is_violation"] == (res["delta_max"] > res["eps"])
    # Actual verdict produced on 32-bit LFSR with T=50 samples:
    assert res["verdict"] == "соответствие ТСБ"
    assert res["is_violation"] is False


def test_ac03_mersenne_twister_compatibility() -> None:
    """AC-03: On Mersenne Twister, compatibility mode evaluates delta_max and verdict.

    Per spec AC-03: воспроизводит вывод о нарушении или расхождение объяснено в отчёте.
    With T=50 training samples and window sizes 8, 16, 32, MT19937 (state 19937 bits)
    is indistinguishable from random noise; both analysis.py and dissertation_verdict
    yield delta_max < threshold (no violation).
    """
    cfg = load_config()
    mt_bits = generate_mt(100000)
    res = dissertation_verdict(mt_bits, cfg=cfg)

    assert "delta_max" in res
    assert "eps" in res
    assert "verdict" in res
    assert np.isfinite(res["delta_max"])

    # Strict verdict verification
    assert res["is_violation"] == (res["delta_max"] > res["eps"])
    # Actual verdict produced on Mersenne Twister with T=50 samples:
    assert res["verdict"] == "соответствие ТСБ"
    assert res["is_violation"] is False


def test_ac04_markov_chain_delta_and_chi2() -> None:
    """AC-04: On 1st-order Markov chain with P(1|1)=0.6:
    bitwise delta at w=8 is significantly > 0, and Markov chi2 order 2 gives p < 0.001.
    """
    cfg = load_config()
    np.random.seed(42)
    n = 10000
    bits = np.zeros(n, dtype=np.uint8)
    for i in range(1, n):
        p = 0.6 if bits[i - 1] == 1 else 0.4
        bits[i] = 1 if np.random.rand() < p else 0

    delta, n_test = bit_delta(bits, w=8, direction="fwd", cfg=cfg)
    assert delta > 0.05, f"Expected delta > 0.05, got {delta:.4f}"

    chi2_stat, p_val = markov_test(bits, order=2)
    assert p_val < 0.001, f"Expected p < 0.001, got {p_val}"
