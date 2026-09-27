"""CLI entry point for stage 0 experiment."""
from __future__ import annotations

import logging

from voiceguard.experiments.stage0 import run_stage0
from voiceguard.fusion.calibrator import LLRCalibrator, build_passport

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = run_stage0(
        manifest_path="data/manifest.parquet",
        data_dir="data",
        models_dir="data/models",
        reports_dir="reports/stage0",
        channels=["clean", "amrnb_12.2"],
    )

    # Build passport
    cal = LLRCalibrator()
    passport_path = build_passport(
        metrics=metrics,
        calibrator=cal,
        sprt_params={"sprt_alpha": 0.001, "sprt_beta": 0.05, "t_max_s": 30.0},
        model_path="data/models/passport.json",
    )
    print("Stage0 done. Report: reports/stage0/report.md")
    print(f"Passport: {passport_path}")
