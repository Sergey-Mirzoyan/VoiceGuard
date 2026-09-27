"""Calibration and evaluation script for VoiceGuard MVP.

Calculates s1 and s2 on calib (up to 300 segments),
trains LLRCalibrator on (s1, s2) and on s1 only.
Evaluates on test split (>= 20 files, balanced live/spoof):
- seg AUC for s1 and s2
- mean s1 / s2 for live and spoof
- SPRT clip-level verdicts for both calibrators.
Saves the better calibrator to data/models/d1_calibrator.joblib.
Updates data/models/passport.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

from voiceguard.config import load_config
from voiceguard.d1.detector import predict_logit
from voiceguard.d2.detector import Reference, _compute_delta, _parse_check_id
from voiceguard.dsp.binarize import symbols_A
from voiceguard.dsp.vad import segment as do_segment
from voiceguard.engine.stream_engine import REDUCED_D2_CHECKS
from voiceguard.fusion.calibrator import SPRT, LLRCalibrator
from voiceguard.types import AudioClip, Segment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_segment_scores(
    seg: Segment,
    d1_model: Any,
    d2_ref: Reference,
    cfg: Any,
) -> tuple[float, float]:
    """Compute (s1, s2) for a single segment."""
    # D1
    try:
        s1 = float(predict_logit(seg, d1_model, cfg))
    except Exception as exc:
        logger.debug("D1 scoring error: %s", exc)
        s1 = 0.0

    # D2 (reduced check set)
    try:
        bits = symbols_A(seg, cfg=cfg, frames="all")
        z_vals: list[float] = []
        for cid in REDUCED_D2_CHECKS:
            _, mode, w, direction = _parse_check_id(cid)
            d = _compute_delta(
                bits,
                w=w,
                mode=mode,
                direction=direction,
                max_train=cfg.d2.max_train,
                train_frac=cfg.d2.train_frac,
                seed=cfg.d2.seed,
            )
            if np.isnan(d):
                continue
            mu0 = d2_ref.mu0.get(cid, 0.0)
            sigma0 = max(d2_ref.sigma0.get(cid, 1.0), 1e-9)
            z = (d - mu0) / sigma0
            z_vals.append(abs(z))
        s2 = float(max(z_vals, default=0.0))
    except Exception as exc:
        logger.debug("D2 scoring error: %s", exc)
        s2 = 0.0

    return s1, s2


def main() -> None:
    cfg = load_config()
    data_dir = Path("data")
    manifest_path = data_dir / "manifest.parquet"
    models_dir = Path("data/models")

    df = pd.read_parquet(manifest_path)
    logger.info("Loaded manifest: %d clips total", len(df))

    d1_model = joblib.load(models_dir / "d1_nb.joblib")
    d2_ref = Reference.load(models_dir / "d2_ref_clean.json")
    logger.info("Loaded D1 model and D2 reference (clean)")

    # -----------------------------------------------------------------------
    # 1. Collect segments from calib (up to 300 segments, balanced)
    # -----------------------------------------------------------------------
    calib_df = df[df["split"] == "calib"]
    calib_live_df = calib_df[calib_df["label"] == "live"]
    calib_spoof_df = calib_df[calib_df["label"] == "spoof"]

    calib_segs_live: list[Segment] = []
    for _, row in calib_live_df.iterrows():
        p = data_dir / row["path"]
        samples, sr = sf.read(str(p))
        clip = AudioClip(samples=samples.astype(np.float32), sr=sr, clip_id=row["clip_id"])
        segs = do_segment(clip, cfg)
        calib_segs_live.extend(segs)
        if len(calib_segs_live) >= 150:
            calib_segs_live = calib_segs_live[:150]
            break

    calib_segs_spoof: list[Segment] = []
    for _, row in calib_spoof_df.iterrows():
        p = data_dir / row["path"]
        samples, sr = sf.read(str(p))
        clip = AudioClip(samples=samples.astype(np.float32), sr=sr, clip_id=row["clip_id"])
        segs = do_segment(clip, cfg)
        calib_segs_spoof.extend(segs)
        if len(calib_segs_spoof) >= 150:
            calib_segs_spoof = calib_segs_spoof[:150]
            break

    all_calib_segs = calib_segs_live + calib_segs_spoof
    all_calib_labels = [0] * len(calib_segs_live) + [1] * len(calib_segs_spoof)
    logger.info(
        "Calib segments: %d total (%d live, %d spoof)",
        len(all_calib_segs),
        len(calib_segs_live),
        len(calib_segs_spoof),
    )

    # Compute scores in parallel
    logger.info("Computing calib scores with joblib (4 workers)...")
    calib_results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(compute_segment_scores)(seg, d1_model, d2_ref, cfg)
        for seg in all_calib_segs
    )
    calib_s1 = [r[0] for r in calib_results]
    calib_s2 = [r[1] for r in calib_results]

    # -----------------------------------------------------------------------
    # 2. Fit calibrators
    # -----------------------------------------------------------------------
    cal_s1 = LLRCalibrator()
    cal_s1.fit(calib_s1, labels=all_calib_labels)
    logger.info("Fitted cal_s1: %s", cal_s1.to_dict())

    cal_s1_s2 = LLRCalibrator()
    cal_s1_s2.fit(calib_s1, calib_s2, labels=all_calib_labels)
    logger.info("Fitted cal_s1_s2: %s", cal_s1_s2.to_dict())

    # -----------------------------------------------------------------------
    # 3. Evaluate on test (24 files: 12 live, 12 spoof)
    # -----------------------------------------------------------------------
    test_df = df[df["split"] == "test"]
    test_live = test_df[test_df["label"] == "live"].head(12)
    test_spoof = test_df[test_df["label"] == "spoof"].head(12)
    test_subset = pd.concat([test_live, test_spoof], ignore_index=True)

    logger.info("Test subset: %d clips (12 live, 12 spoof)", len(test_subset))

    clip_eval_rows: list[dict[str, Any]] = []
    test_seg_s1: list[float] = []
    test_seg_s2: list[float] = []
    test_seg_y: list[int] = []

    for _, row in test_subset.iterrows():
        p = data_dir / row["path"]
        samples, sr = sf.read(str(p))
        clip_id = row["clip_id"]
        true_label = row["label"]
        clip = AudioClip(samples=samples.astype(np.float32), sr=sr, clip_id=clip_id)
        segs = do_segment(clip, cfg)

        sprt_s1 = SPRT(
            alpha=cfg.fusion.sprt_alpha, beta=cfg.fusion.sprt_beta, t_max_s=cfg.fusion.t_max_s
        )
        sprt_s1_s2 = SPRT(
            alpha=cfg.fusion.sprt_alpha, beta=cfg.fusion.sprt_beta, t_max_s=cfg.fusion.t_max_s
        )

        clip_s1s: list[float] = []
        clip_s2s: list[float] = []

        for seg in segs:
            s1, s2 = compute_segment_scores(seg, d1_model, d2_ref, cfg)
            test_seg_s1.append(s1)
            test_seg_s2.append(s2)
            test_seg_y.append(1 if true_label == "spoof" else 0)
            clip_s1s.append(s1)
            clip_s2s.append(s2)

            llr1 = cal_s1.llr(s1)
            sprt_s1.update(llr1)

            llr12 = cal_s1_s2.llr(s1, s2)
            sprt_s1_s2.update(llr12)

        clip_eval_rows.append(
            {
                "clip_id": clip_id,
                "label": true_label,
                "n_segs": len(segs),
                "avg_s1": float(np.mean(clip_s1s)) if clip_s1s else 0.0,
                "avg_s2": float(np.mean(clip_s2s)) if clip_s2s else 0.0,
                "verdict_s1": sprt_s1.verdict,
                "verdict_s1_s2": sprt_s1_s2.verdict,
                "lambda_s1": float(sprt_s1.Lambda),
                "lambda_s1_s2": float(sprt_s1_s2.Lambda),
            }
        )

    # -----------------------------------------------------------------------
    # 4. Compute metrics
    # -----------------------------------------------------------------------
    y_arr = np.array(test_seg_y)
    s1_arr = np.array(test_seg_s1)
    s2_arr = np.array(test_seg_s2)

    auc_s1 = float(roc_auc_score(y_arr, s1_arr)) if len(np.unique(y_arr)) > 1 else 0.5
    auc_s2 = float(roc_auc_score(y_arr, s2_arr)) if len(np.unique(y_arr)) > 1 else 0.5

    live_mask = y_arr == 0
    spoof_mask = y_arr == 1

    mean_s1_live = float(np.mean(s1_arr[live_mask]))
    mean_s1_spoof = float(np.mean(s1_arr[spoof_mask]))
    mean_s2_live = float(np.mean(s2_arr[live_mask]))
    mean_s2_spoof = float(np.mean(s2_arr[spoof_mask]))

    # Clip-level accuracy for SPRT
    def calc_sprt_stats(rows: list[dict[str, Any]], verdict_key: str) -> dict[str, Any]:
        live_rows = [r for r in rows if r["label"] == "live"]
        spoof_rows = [r for r in rows if r["label"] == "spoof"]

        # false alarm: live classified as suspicious
        fa = sum(1 for r in live_rows if r[verdict_key] == "suspicious")
        # detection: spoof classified as suspicious
        det = sum(1 for r in spoof_rows if r[verdict_key] == "suspicious")
        # normal on live:
        live_norm = sum(1 for r in live_rows if r[verdict_key] == "normal")
        # correct = live_norm + det
        corr = live_norm + det
        tot = len(rows)

        return {
            "accuracy": corr / tot if tot else 0.0,
            "fa_rate": fa / len(live_rows) if live_rows else 0.0,
            "detection_rate": det / len(spoof_rows) if spoof_rows else 0.0,
            "live_normal": live_norm,
            "spoof_suspicious": det,
            "total_clips": tot,
        }

    stats_s1 = calc_sprt_stats(clip_eval_rows, "verdict_s1")
    stats_s1_s2 = calc_sprt_stats(clip_eval_rows, "verdict_s1_s2")

    print("\n=======================================================")
    print("                 РЕЗУЛЬТАТЫ СРАВНЕНИЯ                  ")
    print("=======================================================")
    print(f"Количество тестовых сегментов: {len(test_seg_y)}")
    print(f"AUC s1 (D1): {auc_s1:.4f}")
    print(f"AUC s2 (D2): {auc_s2:.4f}")
    print(f"Средний s1: live={mean_s1_live:.3f}, spoof={mean_s1_spoof:.3f}")
    print(f"Средний s2: live={mean_s2_live:.3f}, spoof={mean_s2_spoof:.3f}")
    print("\nСтатистика SPRT на 24 тестовых клипах:")
    print(
        f"  Калибратор (s1 only):  Точность={stats_s1['accuracy']*100:.1f}%, "
        f"Det={stats_s1['detection_rate']*100:.1f}%, FA={stats_s1['fa_rate']*100:.1f}%"
    )
    print(
        f"  Калибратор (s1 + s2):  Точность={stats_s1_s2['accuracy']*100:.1f}%, "
        f"Det={stats_s1_s2['detection_rate']*100:.1f}%, FA={stats_s1_s2['fa_rate']*100:.1f}%"
    )
    print(
        f"{'Clip ID':<12} | {'Label':<6} | {'Segs':<4} | {'s1 avg':<7} | "
        f"{'s2 avg':<7} | {'SPRT (s1)':<10} | {'SPRT (s1+s2)':<12}"
    )
    print("-" * 75)
    for r in clip_eval_rows:
        print(
            f"{r['clip_id']:<12} | {r['label']:<6} | {r['n_segs']:<4} | {r['avg_s1']:<7.2f} | "
            f"{r['avg_s2']:<7.2f} | {r['verdict_s1']:<10} | {r['verdict_s1_s2']:<12}"
        )

    # -----------------------------------------------------------------------
    # 5. Decide which calibrator to save
    # -----------------------------------------------------------------------
    # Better if higher accuracy, or equal accuracy with lower FA or higher detection
    s2_coef = cal_s1_s2.to_dict().get("coef", [0.0, 0.0])[1] if cal_s1_s2._clf else 0.0
    use_s1_s2 = (
        stats_s1_s2["accuracy"] > stats_s1["accuracy"]
        or (
            stats_s1_s2["accuracy"] == stats_s1["accuracy"]
            and stats_s1_s2["fa_rate"] <= stats_s1["fa_rate"]
            and stats_s1_s2["detection_rate"] >= stats_s1["detection_rate"]
            and s2_coef > 0.01
        )
    )

    calibrator_path = models_dir / "d1_calibrator.joblib"
    passport_path = models_dir / "passport.json"

    if use_s1_s2:
        chosen_cal = cal_s1_s2
        chosen_name = "(s1, s2)"
        d2_contribution_note = f"D2 дал положительный вклад (coef={s2_coef:.4f}, AUC={auc_s2:.4f})"
    else:
        chosen_cal = cal_s1
        chosen_name = "(s1 only)"
        d2_contribution_note = (
            f"D2 не дал прироста качества (AUC D2={auc_s2:.4f} vs D1={auc_s1:.4f}); "
            f"сохранён надёжный калибратор s1"
        )

    chosen_cal.save(calibrator_path)
    logger.info("Saved %s calibrator to %s", chosen_name, calibrator_path)

    # -----------------------------------------------------------------------
    # 6. Build and save passport
    # -----------------------------------------------------------------------
    passport_data = {
        "calibrator_type": chosen_name,
        "calibrator_params": chosen_cal.to_dict(),
        "d2_note": d2_contribution_note,
        "metrics": {
            "clean": {
                "d1_auc": auc_s1,
                "d2_auc": auc_s2,
                "mean_s1_live": mean_s1_live,
                "mean_s1_spoof": mean_s1_spoof,
                "mean_s2_live": mean_s2_live,
                "mean_s2_spoof": mean_s2_spoof,
                "test_clips": len(test_subset),
                "test_segments": len(test_seg_y),
                "sprt_s1": stats_s1,
                "sprt_s1_s2": stats_s1_s2,
            }
        },
        "sprt": {
            "alpha": cfg.fusion.sprt_alpha,
            "beta": cfg.fusion.sprt_beta,
            "t_max_s": cfg.fusion.t_max_s,
        },
    }

    with open(passport_path, "w", encoding="utf-8") as f:
        json.dump(passport_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved updated passport to %s", passport_path)

    print("\n=======================================================")
    print(f"Выбран калибратор: {chosen_name}")
    print(f"Обоснование: {d2_contribution_note}")
    print(f"Паспорт обновлён: {passport_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
