from __future__ import annotations

import numpy as np
from scipy import stats

from voiceguard.config import Config, load_config
from voiceguard.d2.chi2 import markov_test
from voiceguard.d2.predictors import run_checks
from voiceguard.d2.reference import Reference
from voiceguard.dsp.binarize import symbols_A
from voiceguard.dsp.pitch import jitter_sequence, periods, symbols_B
from voiceguard.types import D2Result, Segment


def threshold(cfg: Config | None = None, m: int = 12) -> float:
    """Compute single-test rejection threshold z_{alpha / (2m)} (FR-08).

    For alpha = 0.01 and m = 12, this is approximately 3.34.
    """
    if cfg is None:
        cfg = load_config()
    alpha = cfg.d2.alpha
    return float(stats.norm.isf(alpha / (2.0 * max(1, m))))


def analyze(
    segment: Segment,
    ref: Reference,
    cfg: Config | None = None,
) -> D2Result:
    """Analyze a single 1.0 s speech segment against the live speech reference model (FR-08).

    Computes standardized deviations:
        Z_j = (delta_j - mu0_j) / sigma0_j
    and overall test statistic:
        s2 = max_j |Z_j|

    Returns:
        D2Result: Populated with deltas, Z-scores, s2, chi2 p-values, and metadata.
    """
    if cfg is None:
        cfg = load_config()

    variant = ref.variant
    if variant == "A":
        bits = symbols_A(segment, cfg=cfg, frames=ref.frames)  # type: ignore[arg-type]
        d_seq = None
    else:
        p_seq = periods(segment)
        d_seq = jitter_sequence(p_seq)
        bits = symbols_B(p_seq)

    delta_dict: dict[str, float] = {}
    z_dict: dict[str, float] = {}

    if len(bits) > 0:
        checks = run_checks(bits, variant=variant, cfg=cfg, d_seq=d_seq)
        use_robust = cfg.d2.reference.robust

        for cid, (d_val, _) in checks.items():
            delta_dict[cid] = float(d_val)
            if np.isnan(d_val):
                z_dict[cid] = float("nan")
                continue

            if use_robust:
                center = ref.median0.get(cid, 0.0)
                scale = ref.mad0.get(cid, 1e-4) or 1e-4
            else:
                center = ref.mu0.get(cid, 0.0)
                scale = ref.sigma0.get(cid, 1e-4) or 1e-4

            z_val = (d_val - center) / scale
            z_dict[cid] = float(z_val)

    valid_z = [abs(z) for z in z_dict.values() if not np.isnan(z)]
    s2 = float(max(valid_z)) if valid_z else 0.0

    # Markov chi-square tests
    chi2_p: dict[str, float] = {}
    for order in cfg.d2.chi2_orders:
        order_key = str(order)
        ref_p = np.array(ref.pi0[order_key]) if order_key in ref.pi0 else None
        _, p_val = markov_test(bits, order=order, ref_probs=ref_p)
        chi2_p[f"order_{order}"] = float(p_val)

    return D2Result(
        delta=delta_dict,
        z=z_dict,
        s2=s2,
        chi2_p=chi2_p,
        n_symbols=len(bits),
        variant=variant,
    )
