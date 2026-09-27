from __future__ import annotations

import numpy as np
from scipy import stats


def markov_test(
    bits: np.ndarray,
    order: int,
    ref_probs: np.ndarray | None = None,
) -> tuple[float, float]:
    """Markov transition chi-square test for binary sequences (FR-05).

    Tests transitions from order-bit contexts to the next bit.
    Laplace smoothing (+1) is added to each cell as in dissertation.

    Args:
        bits: 1D array of 0/1 bits.
        order: Context length in bits.
        ref_probs: Optional reference conditional probabilities pi0(b|c) of shape (2^order, 2).
                   If None, assumes uniform pi0(b|c) = 0.5 (PRNG mode).

    Returns:
        tuple[float, float]: (chi2_statistic, p_value).
    """
    b_arr = np.asarray(bits, dtype=np.uint8)
    n = len(b_arr)
    n_contexts = 1 << order
    df = n_contexts

    # Counts matrix with Laplace smoothing (+1)
    observed = np.ones((n_contexts, 2), dtype=np.float64)

    if n > order:
        w_len = order + 1
        windows = np.lib.stride_tricks.sliding_window_view(b_arr, w_len)
        powers = 1 << np.arange(order)[::-1]
        contexts = windows[:, :order].dot(powers)
        next_bits = windows[:, order]

        for c, b in zip(contexts, next_bits, strict=True):
            observed[c, b] += 1.0

    n_c = observed.sum(axis=1)  # shape (2^order,)

    if ref_probs is None:
        pi0 = np.full((n_contexts, 2), 0.5, dtype=np.float64)
    else:
        pi0 = np.asarray(ref_probs, dtype=np.float64)
        if pi0.shape != (n_contexts, 2):
            raise ValueError(f"ref_probs must have shape ({n_contexts}, 2), got {pi0.shape}")

    expected = n_c[:, None] * pi0
    # Avoid division by zero
    valid = expected > 0.0
    chi2_stat = float(np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]))
    p_val = float(stats.chi2.sf(chi2_stat, df=df))

    return chi2_stat, p_val
