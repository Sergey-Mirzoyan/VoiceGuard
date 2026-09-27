from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from voiceguard.dsp.lpc import levinson_durbin, lpc_frame, residual


def test_ac01_levinson_durbin_matches_solve_toeplitz() -> None:
    """AC-01: levinson_durbin matches solve_toeplitz to 1e-6."""
    np.random.seed(42)
    # Generate random positive-definite autocorrelation vector
    for order in (10, 16):
        x = np.random.randn(1000)
        r = signal.correlate(x, x, mode="full")[len(x) - 1 : len(x) + order]
        # Regularization matching FR-02
        r_reg = r.copy()
        r_reg[0] *= 1.0 + 1e-9

        a_ld, _ = levinson_durbin(r, order)
        # solve_toeplitz solves T * a = r[1:order+1] where T is Toeplitz with row/col r[:order]
        a_toeplitz = linalg.solve_toeplitz((r_reg[:order], r_reg[:order]), r_reg[1 : order + 1])

        max_diff = float(np.max(np.abs(a_ld - a_toeplitz)))
        assert max_diff < 1e-6, f"Order {order}: max diff {max_diff:.2e} >= 1e-6"


def test_ac02_ar2_parameter_estimation() -> None:
    """AC-02: AR(2) process estimated with error <= 0.05 on 10,000 samples."""
    np.random.seed(123)
    n_samples = 10000
    a_true = np.array([0.6, -0.3], dtype=np.float64)  # x[n] = 0.6 x[n-1] - 0.3 x[n-2] + e[n]

    # Generate AR(2) time series
    white_noise = np.random.randn(n_samples)
    # scipy.signal.lfilter with denominator [1, -0.6, 0.3]
    x = signal.lfilter([1.0], [1.0, -a_true[0], -a_true[1]], white_noise)

    # Compute autocorrelation
    r = signal.correlate(x, x, mode="full")[len(x) - 1 : len(x) + 2]
    a_est, _ = levinson_durbin(r, order=2)

    err = np.abs(a_est - a_true)
    assert np.all(err < 0.05), f"AR(2) parameter estimation error {err} exceeds 0.05 (est: {a_est})"


def test_ac03_residual_autocorrelation_white_noise() -> None:
    """AC-03: Residual of white noise through AR filter has |rho| < 0.05 at lags 1..10."""
    np.random.seed(999)
    sr = 8000
    n_samples = 24000  # 3 seconds
    white_noise = np.random.randn(n_samples)

    # Filter with known AR polynomial (e.g. order 4 stable all-pole filter)
    # Poles inside unit circle
    ar_coeffs = np.array([0.5, -0.2, 0.1, -0.05], dtype=np.float64)
    den = np.concatenate([[1.0], -ar_coeffs])
    x = signal.lfilter([1.0], den, white_noise).astype(np.float32)

    # Compute inverse-filtered residual
    res = residual(x, sr=sr)

    # Autocorrelation of the residual
    # Discard transient initial frame
    steady_res = res[sr // 2 :]
    steady_res = steady_res - np.mean(steady_res)
    corr = signal.correlate(steady_res, steady_res, mode="full")
    mid = len(steady_res) - 1
    r_lags = corr[mid : mid + 11]
    rho = r_lags[1:] / r_lags[0]

    max_rho = float(np.max(np.abs(rho)))
    assert max_rho < 0.05, f"Residual autocorrelation lags 1..10 max |rho| = {max_rho:.4f} >= 0.05"


def test_lpc_silence_and_edge_cases() -> None:
    """Test LPC handling of silence and degenerate input."""
    silence = np.zeros(160, dtype=np.float32)
    a = lpc_frame(silence, order=10)
    assert len(a) == 10
    assert np.all(a == 0.0)

    res = residual(silence, sr=8000)
    assert len(res) == len(silence)
    assert np.all(res == 0.0)

    # Empty array
    assert len(residual(np.zeros(0, dtype=np.float32), sr=8000)) == 0
