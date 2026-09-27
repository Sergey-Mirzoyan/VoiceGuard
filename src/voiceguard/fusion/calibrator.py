"""Fusion module: LLR calibration, SPRT decision, passport (ЧТЗ-08).

FR-02: LLR calibrator (logistic regression on [s1, s2]).
FR-03: SPRT (Wald sequential test).
FR-07: Passport JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLR Calibrator
# ---------------------------------------------------------------------------


class LLRCalibrator:
    """Logistic-regression-based LLR calibrator.

    Trained on calib set to produce log-likelihood-ratio from (s1, s2) scores.
    """

    def __init__(self) -> None:
        self._clf: Any = None

    def fit(
        self,
        s1_scores: list[float],
        s2_scores: list[float],
        labels: list[int],
        seed: int = 42,
    ) -> "LLRCalibrator":
        """Fit the calibrator on calibration data.

        Args:
            s1_scores: D1 logits.
            s2_scores: D2 s2 values.
            labels: 0=live, 1=spoof.
            seed: Random seed.
        """
        from sklearn.linear_model import LogisticRegression

        X = np.column_stack([s1_scores, s2_scores])
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            logger.warning("LLRCalibrator: only one class in training data, using trivial calibrator")
            self._clf = None
            return self

        clf = LogisticRegression(max_iter=300, random_state=seed)
        clf.fit(X, y)
        self._clf = clf
        return self

    def llr(self, s1: float, s2: float) -> float:
        """Compute log-likelihood-ratio for a single (s1, s2) pair."""
        if self._clf is None:
            # Fallback: simple average of scores
            return float(s1 + s2) / 2

        X = np.array([[s1, s2]])
        prob = self._clf.predict_proba(X)[0]  # [p_live, p_spoof]
        p_spoof = float(prob[1])
        p_live = float(prob[0])
        p_spoof = np.clip(p_spoof, 1e-9, 1 - 1e-9)
        p_live = np.clip(p_live, 1e-9, 1 - 1e-9)
        return float(np.log(p_spoof / p_live))

    def to_dict(self) -> dict[str, Any]:
        if self._clf is None:
            return {"type": "trivial"}
        coef = self._clf.coef_[0].tolist()
        intercept = float(self._clf.intercept_[0])
        return {"type": "logreg", "coef": coef, "intercept": intercept}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self, path)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LLRCalibrator":
        import joblib

        return joblib.load(path)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# SPRT
# ---------------------------------------------------------------------------


class SPRT:
    """Wald sequential probability ratio test (FR-03).

    Each call to `update(llr)` accumulates the log-likelihood ratio.
    Verdict: "suspicious" | "normal" | "pending" | "undecided".

    Args:
        alpha: Target false alarm probability.
        beta: Target missed detection probability.
        t_max_s: Maximum test duration in seconds (after which "undecided").
        sr_segments: Number of segments per second (default 1.0).
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta: float = 0.05,
        t_max_s: float = 30.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.t_max_s = t_max_s

        # Thresholds
        self.A = np.log((1 - beta) / alpha)   # upper: suspicious
        self.B = np.log(beta / (1 - alpha))   # lower: normal

        self._lambda: float = 0.0
        self._t: float = 0.0
        self._verdict: str = "pending"

    @property
    def Lambda(self) -> float:
        return self._lambda

    @property
    def t(self) -> float:
        return self._t

    @property
    def verdict(self) -> str:
        return self._verdict

    def update(self, llr: float, dt: float = 1.0) -> str:
        """Add one segment's LLR and return current verdict.

        Args:
            llr: Log-likelihood ratio for this segment.
            dt: Duration of this segment in seconds.

        Returns:
            Current verdict string.
        """
        if self._verdict not in ("pending",):
            return self._verdict

        self._lambda += llr
        self._t += dt

        if self._lambda >= self.A:
            self._verdict = "suspicious"
        elif self._lambda <= self.B:
            self._verdict = "normal"
        elif self._t >= self.t_max_s:
            self._verdict = "undecided"
        else:
            self._verdict = "pending"

        return self._verdict

    def reset(self) -> None:
        """Reset state for a new session."""
        self._lambda = 0.0
        self._t = 0.0
        self._verdict = "pending"


# ---------------------------------------------------------------------------
# Passport
# ---------------------------------------------------------------------------


def build_passport(
    metrics: dict[str, Any],
    calibrator: LLRCalibrator,
    sprt_params: dict[str, float],
    model_path: str | Path = "data/models/passport.json",
) -> Path:
    """Build and save detector passport JSON.

    Args:
        metrics: Stage-0 metrics dict.
        calibrator: Fitted LLRCalibrator.
        sprt_params: Dict with sprt_alpha, sprt_beta, t_max_s.
        model_path: Output path.

    Returns:
        Path to passport.json.
    """
    passport: dict[str, Any] = {
        "version": "1.0",
        "description": "VoiceGuard AM detector passport",
        "sprt": sprt_params,
        "calibrator": calibrator.to_dict(),
        "metrics_by_channel": {},
        "notes": [
            "D1: LFCC + LogisticRegression",
            "D2: LPC residual binarization, logistic predictor, variant A",
            "Fusion: LogisticRegression on (s1, s2)",
            "Data: Hy-Generated, 300 live + 300 spoof clips",
        ],
    }

    for ch, m in metrics.items():
        passport["metrics_by_channel"][ch] = {
            "d1_auc": round(m.get("d1_auc", 0), 4),
            "d1_eer": round(m.get("d1_eer", 0), 4),
            "d2_auc": round(m.get("d2_auc", 0), 4),
            "d2_eer": round(m.get("d2_eer", 0), 4),
            "fusion_auc": round(m.get("fusion_auc", 0), 4),
            "fusion_eer": round(m.get("fusion_eer", 0), 4),
            "n_test": m.get("n_test", 0),
        }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2, ensure_ascii=False)

    logger.info("Passport saved to %s", model_path)
    return model_path
