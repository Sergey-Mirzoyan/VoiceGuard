"""Stage 0 experiment runner (упрощённое ЧТЗ-07).

Builds D2 references on calib set, evaluates D1+D2 on test,
computes AUC and EER, checks gates G1, writes reports/stage0/report.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # EER is where FPR ≈ FNR
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


def run_stage0(
    manifest_path: str | Path = "data/manifest.parquet",
    data_dir: str | Path = "data",
    models_dir: str | Path = "data/models",
    reports_dir: str | Path = "reports/stage0",
    channels: list[str] | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Run stage 0: train D1, build D2 references, evaluate on test, write report.

    Returns:
        Dictionary with all metrics.
    """
    from voiceguard.config import load_config

    if cfg is None:
        cfg = load_config()
    if channels is None:
        channels = ["clean", "amrnb_12.2"]

    import pandas as pd
    import soundfile as sf

    from voiceguard.d1.detector import load_model, predict_logit, train
    from voiceguard.d2.detector import Reference, build_reference, score_segment
    from voiceguard.dsp.vad import segment as do_segment
    from voiceguard.types import AudioClip

    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(manifest_path)

    def _load_clip(row: pd.Series) -> AudioClip:
        path = data_dir / row["path"]
        samples, sr = sf.read(str(path))
        samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return AudioClip(
            samples=samples, sr=sr, clip_id=row["clip_id"], meta={"label": row["label"]}
        )

    def _clips_for_split(split: str, label: str | None = None) -> list[tuple[AudioClip, str]]:
        sub = df[df["split"] == split]
        if label:
            sub = sub[sub["label"] == label]
        result = []
        for _, row in sub.iterrows():
            try:
                clip = _load_clip(row)
                result.append((clip, row["label"]))
            except Exception as e:
                logger.debug("Failed to load %s: %s", row["path"], e)
        return result

    def _segs_for_split(split: str, label: str | None = None) -> list[tuple[Any, str]]:
        clips = _clips_for_split(split, label)
        result = []
        for clip, lbl in clips:
            segs = do_segment(clip, cfg)
            for s in segs:
                result.append((s, lbl))
        return result

    # -----------------------------------------------------------------------
    # 1. Train D1
    # -----------------------------------------------------------------------
    d1_model_path = models_dir / "d1_nb.joblib"
    if d1_model_path.exists():
        logger.info("Loading existing D1 model from %s", d1_model_path)
        d1_model = load_model(d1_model_path)
    else:
        logger.info("Training D1 on train split...")
        train_pairs = _segs_for_split("train")
        if not train_pairs:
            raise RuntimeError("No training segments found")
        train_segs = [p[0] for p in train_pairs]
        train_labels = [p[1] for p in train_pairs]
        d1_model = train(
            train_segs,
            train_labels,
            cfg=cfg,
            aug_channels=["clean", "g711a", "amrnb_12.2"],
            model_path=d1_model_path,
        )

    # -----------------------------------------------------------------------
    # 2. Build D2 references on calib split (live only)
    # -----------------------------------------------------------------------
    refs: dict[str, Reference] = {}
    for ch in channels:
        ref_path = models_dir / f"d2_ref_{ch.replace('.', '_')}.json"
        if ref_path.exists():
            logger.info("Loading existing D2 reference for %s", ch)
            refs[ch] = Reference.load(ref_path)
            continue

        logger.info("Building D2 reference for channel '%s' on calib live segments...", ch)
        calib_live = _segs_for_split("calib", "live")
        if not calib_live:
            logger.warning("No calib live segments for channel %s, skipping", ch)
            continue
        calib_segs_live = [p[0] for p in calib_live]

        # Apply channel if not clean
        if ch != "clean":
            from voiceguard.channel.simulator import apply
            from voiceguard.channel.spec import ChannelSpec

            spec = ChannelSpec.parse(ch)
            aug_segs = []
            for s in calib_segs_live:
                try:
                    clip = AudioClip(samples=s.samples, sr=s.sr, clip_id=s.clip_id)
                    aug_clip = apply(clip, spec, seed=42)
                    new_segs = do_segment(aug_clip, cfg)
                    aug_segs.extend(new_segs)
                except Exception as e:
                    logger.debug("Channel apply failed: %s", e)
                    aug_segs.append(s)
            calib_segs_live = aug_segs if aug_segs else calib_segs_live

        ref = build_reference(calib_segs_live, channel=ch, cfg=cfg, frames="all")
        ref.save(ref_path)
        refs[ch] = ref

    # -----------------------------------------------------------------------
    # 3. Evaluate on test
    # -----------------------------------------------------------------------
    test_pairs = _segs_for_split("test")
    if not test_pairs:
        logger.warning("No test segments found, metrics will be empty")

    metrics: dict[str, Any] = {}

    for ch in channels:
        ref_eval = refs.get(ch)
        if ref_eval is None:
            continue

        logger.info("Evaluating on channel '%s', %d test segments", ch, len(test_pairs))

        s1_scores: list[float] = []
        s2_scores: list[float] = []
        y_true: list[int] = []

        for seg, lbl in test_pairs:
            y_val = 1 if lbl == "spoof" else 0
            y_true.append(y_val)
            try:
                s1 = predict_logit(seg, d1_model, cfg)
                s1_scores.append(s1)
            except Exception:
                s1_scores.append(0.0)
            try:
                d2 = score_segment(seg, ref_eval, cfg)
                s2_scores.append(d2.s2)
            except Exception:
                s2_scores.append(0.0)

        if not y_true:
            continue

        yt = np.array(y_true)
        s1a = np.array(s1_scores)
        s2a = np.array(s2_scores)

        d1_auc = _auc(yt, s1a) if len(np.unique(yt)) > 1 else 0.5
        d1_eer = _eer(yt, s1a) if len(np.unique(yt)) > 1 else 0.5
        d2_auc = _auc(yt, s2a) if len(np.unique(yt)) > 1 else 0.5
        d2_eer = _eer(yt, s2a) if len(np.unique(yt)) > 1 else 0.5

        # Fusion D1+D2 via logistic regression on calib
        calib_pairs = _segs_for_split("calib")
        fusion_auc, fusion_eer = _fusion_metrics(
            calib_pairs, test_pairs, d1_model, ref_eval, cfg, ch
        )

        metrics[ch] = {
            "d1_auc": d1_auc,
            "d1_eer": d1_eer,
            "d2_auc": d2_auc,
            "d2_eer": d2_eer,
            "fusion_auc": fusion_auc,
            "fusion_eer": fusion_eer,
            "n_test": len(y_true),
        }

        logger.info(
            "Channel %s: D1 AUC=%.3f EER=%.3f | D2 AUC=%.3f EER=%.3f | D1+D2 AUC=%.3f EER=%.3f",
            ch,
            d1_auc,
            d1_eer,
            d2_auc,
            d2_eer,
            fusion_auc,
            fusion_eer,
        )

    # -----------------------------------------------------------------------
    # 4. Write report
    # -----------------------------------------------------------------------
    _write_report(metrics, channels, reports_dir)
    return metrics


def _fusion_metrics(
    calib_pairs: list[Any],
    test_pairs: list[Any],
    d1_model: Any,
    ref: Any,
    cfg: Any,
    channel: str,
) -> tuple[float, float]:
    """Train fusion LR on calib, evaluate on test."""
    from sklearn.linear_model import LogisticRegression

    from voiceguard.d1.detector import predict_logit
    from voiceguard.d2.detector import score_segment

    def extract(pairs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        Xf: list[list[float]] = []
        yf: list[int] = []
        for seg, lbl in pairs:
            try:
                s1 = predict_logit(seg, d1_model, cfg)
                d2r = score_segment(seg, ref, cfg)
                s2 = d2r.s2
                Xf.append([s1, s2])
                yf.append(1 if lbl == "spoof" else 0)
            except Exception:
                pass
        return np.array(Xf), np.array(yf)

    X_c, y_c = extract(calib_pairs)
    X_t, y_t = extract(test_pairs)

    if len(X_c) < 4 or len(np.unique(y_c)) < 2:
        return 0.5, 0.5
    if len(X_t) < 2 or len(np.unique(y_t)) < 2:
        return 0.5, 0.5

    try:
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_c, y_c)
        fusion_scores = clf.predict_proba(X_t)[:, 1]
        f_auc = _auc(y_t, fusion_scores)
        f_eer = _eer(y_t, fusion_scores)
        return f_auc, f_eer
    except Exception:
        return 0.5, 0.5


def _write_report(metrics: dict[str, Any], channels: list[str], reports_dir: Path) -> None:
    """Write markdown report with AUC/EER table and G1 gate check."""
    lines: list[str] = [
        "# VoiceGuard AM — Отчёт этапа 0",
        "",
        "**Дата:** авто-сгенерировано",
        "",
        "## Результаты: AUC и EER по каналам",
        "",
        "| Канал | D1 AUC | D1 EER | D2 AUC | D2 EER | D1+D2 AUC | D1+D2 EER | N_test |",
        "|-------|--------|--------|--------|--------|-----------|-----------|--------|",
    ]

    for ch in channels:
        m = metrics.get(ch)
        if m is None:
            lines.append(f"| {ch} | — | — | — | — | — | — | — |")
            continue
        row = (
            f"| {ch} "
            f"| {m['d1_auc']:.3f} "
            f"| {m['d1_eer']:.3f} "
            f"| {m['d2_auc']:.3f} "
            f"| {m['d2_eer']:.3f} "
            f"| {m['fusion_auc']:.3f} "
            f"| {m['fusion_eer']:.3f} "
            f"| {m['n_test']} |"
        )
        lines.append(row)

    lines += [
        "",
        "## Проверка ворот G1",
        "",
        "**Критерий G1:** D1 AUC ≥ 0.65 хотя бы на одном канале.",
        "",
    ]

    g1_any = any(metrics.get(ch, {}).get("d1_auc", 0) >= 0.65 for ch in channels)
    if g1_any:
        lines.append("✅ **G1 ПРОЙДЕНО**: D1 достигает AUC ≥ 0.65")
    else:
        lines.append("❌ **G1 НЕ ПРОЙДЕНО**: D1 AUC < 0.65 на всех каналах")

    lines += [
        "",
        "## Известные ограничения",
        "",
        "- D2 использует линейные предикторы (LogReg), не MLP.",
        "- D1 использует LFCC + LogReg, не LCNN.",
        "- Данные: по 300 клипов на класс из Hy-Generated.",
        "- AMR-NB требует Docker с voiceguard-base для кодека.",
        "",
    ]

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)
