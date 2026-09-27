"""VoiceGuard stream analysis engine (ЧТЗ-09).

Processes audio segments, runs D1 + D2 (background thread for D2),
feeds LLR into SPRT, returns WindowScore per segment.
"""
from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np

from voiceguard.config import Config, load_config
from voiceguard.types import AudioClip, Segment, WindowScore

logger = logging.getLogger(__name__)

# Reduced check set for streaming D2 (as per ЧТЗ-09)
REDUCED_D2_CHECKS = [
    "A_block_w8_fwd",
    "A_block_w16_fwd",
    "A_block_w32_fwd",
    "A_bit_w16_fwd",
]


class StreamEngine:
    """Streaming analysis engine.

    Loads models on construction, processes segments one by one.
    D2 runs in a background thread executor.

    Args:
        cfg: Config object.
        channel: Channel string (e.g. "clean", "g711a", "amrnb_12.2").
        models_dir: Directory with d1_nb.joblib, d2_ref_*.json, d1_calibrator.joblib.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        channel: str = "clean",
        models_dir: str | Path = "data/models",
    ) -> None:
        self.cfg = cfg or load_config()
        self.channel = channel
        self.models_dir = Path(models_dir)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2")

        self._d1_model: Any = None
        self._d2_ref: Any = None
        self._calibrator: Any = None
        self._sprt: Any = None

        self._d2_queue: queue.Queue[Any] = queue.Queue()
        self._pending_d2: dict[int, Any] = {}
        self._t: float = 0.0
        self._seg_idx: int = 0

        self._load_models()
        self._init_sprt()

    def _load_models(self) -> None:
        from voiceguard.d1.detector import load_model

        d1_path = self.models_dir / "d1_nb.joblib"
        if d1_path.exists():
            self._d1_model = load_model(d1_path)
        else:
            logger.warning("D1 model not found at %s", d1_path)

        ch_safe = self.channel.replace(".", "_")
        d2_path = self.models_dir / f"d2_ref_{ch_safe}.json"
        if not d2_path.exists():
            # fallback to clean
            d2_path = self.models_dir / "d2_ref_clean.json"
        if d2_path.exists():
            from voiceguard.d2.detector import Reference

            self._d2_ref = Reference.load(d2_path)
        else:
            logger.warning("D2 reference not found")

        cal_path = self.models_dir / "d1_calibrator.joblib"
        if cal_path.exists():
            from voiceguard.fusion.calibrator import LLRCalibrator

            self._calibrator = LLRCalibrator.load(cal_path)
        else:
            logger.info("No calibrator found, using trivial LLR")
            from voiceguard.fusion.calibrator import LLRCalibrator

            self._calibrator = LLRCalibrator()

    def _init_sprt(self) -> None:
        from voiceguard.fusion.calibrator import SPRT

        self._sprt = SPRT(
            alpha=self.cfg.fusion.sprt_alpha,
            beta=self.cfg.fusion.sprt_beta,
            t_max_s=self.cfg.fusion.t_max_s,
        )

    def _compute_d2_bg(self, seg: Segment, check_ids: list[str]) -> Any | None:
        """Run D2 in background thread with reduced check set."""
        if self._d2_ref is None:
            return None
        try:
            from voiceguard.d2.detector import _compute_delta, _parse_check_id
            from voiceguard.dsp.binarize import symbols_A
            from voiceguard.types import D2Result

            bits = symbols_A(seg, cfg=self.cfg, frames="all")
            deltas = {}
            z_scores = {}
            chi2_p = {}
            for cid in check_ids:
                _, mode, w, direction = _parse_check_id(cid)
                d = _compute_delta(
                    bits,
                    w=w,
                    mode=mode,
                    direction=direction,
                    max_train=self.cfg.d2.max_train,
                    train_frac=self.cfg.d2.train_frac,
                    seed=self.cfg.d2.seed,
                )
                deltas[cid] = d
                if np.isnan(d):
                    z_scores[cid] = float("nan")
                    chi2_p[cid] = float("nan")
                    continue
                mu0 = self._d2_ref.mu0.get(cid, 0.0)
                sigma0 = max(self._d2_ref.sigma0.get(cid, 1.0), 1e-9)
                z = (d - mu0) / sigma0
                z_scores[cid] = z
                chi2_p[cid] = float(2 * np.exp(-0.5 * z**2))  # approx

            s2 = max((abs(v) for v in z_scores.values() if not np.isnan(v)), default=0.0)
            return D2Result(
                delta=deltas,
                z=z_scores,
                s2=s2,
                chi2_p=chi2_p,
                n_symbols=len(bits),
                variant="A",
            )
        except Exception as exc:
            logger.debug("D2 background error: %s", exc)
            return None

    def process_segment(self, seg: Segment) -> WindowScore:
        """Process one 1-second segment, return WindowScore.

        D1 runs synchronously, D2 runs in background executor.
        """
        from voiceguard.d1.detector import predict_logit

        idx = self._seg_idx
        self._seg_idx += 1
        self._t += 1.0

        # D1
        s1: float | None = None
        if self._d1_model is not None:
            try:
                s1 = predict_logit(seg, self._d1_model, self.cfg)
            except Exception as exc:
                logger.debug("D1 error: %s", exc)

        # D2 — submit to background, use previous result if available
        use_reduced = self.cfg.engine.stream_d2_checks == "reduced"
        check_ids = REDUCED_D2_CHECKS if use_reduced else (
            self._d2_ref.check_ids if self._d2_ref else REDUCED_D2_CHECKS
        )
        future = self._executor.submit(self._compute_d2_bg, seg, check_ids)
        self._pending_d2[idx] = future

        # Try to get previous D2 result (non-blocking)
        s2: float | None = None
        for past_idx in list(self._pending_d2.keys()):
            if past_idx < idx:
                f = self._pending_d2[past_idx]
                if f.done():
                    res = f.result()
                    if res is not None:
                        s2 = res.s2
                    del self._pending_d2[past_idx]

        # LLR
        _s1 = s1 if s1 is not None else 0.0
        _s2 = s2 if s2 is not None else 0.0
        llr = self._calibrator.llr(_s1, _s2)

        # SPRT
        verdict = self._sprt.update(llr, dt=1.0)

        return WindowScore(
            t=self._t,
            s1=s1,
            s2=s2,
            llr=llr,
            Lambda=self._sprt.Lambda,
            verdict=verdict,
        )

    def reset(self) -> None:
        """Reset SPRT and internal state for a new call."""
        self._sprt.reset()
        self._t = 0.0
        self._seg_idx = 0
        self._pending_d2.clear()

    def process_audio(
        self,
        samples: np.ndarray,
        sr: int,
        channel_override: str | None = None,
    ) -> list[WindowScore]:
        """Process a full audio array (non-streaming file analysis).

        Applies channel simulation if needed, segments, processes each.
        Returns list of WindowScore per 1-second segment.
        """
        self.reset()

        clip = AudioClip(
            samples=samples.astype(np.float32),
            sr=sr,
            clip_id="analyze",
        )

        # Apply channel
        ch = channel_override or self.channel
        if ch != "clean":
            try:
                from voiceguard.channel.simulator import apply
                from voiceguard.channel.spec import ChannelSpec

                spec = ChannelSpec.parse(ch)
                clip = apply(clip, spec)
            except Exception as exc:
                logger.warning("Channel apply failed: %s, using clean", exc)

        from voiceguard.dsp.vad import segment as do_segment

        segs = do_segment(clip, self.cfg)
        results: list[WindowScore] = []
        for seg in segs:
            ws = self.process_segment(seg)
            results.append(ws)

        # Wait for all pending D2
        for f in self._pending_d2.values():
            try:
                f.result(timeout=5.0)
            except Exception:
                pass
        self._pending_d2.clear()

        return results
