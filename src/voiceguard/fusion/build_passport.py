"""CLI: build passport from stage0 metrics (requires data/models/d1_nb.joblib etc.)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from voiceguard.fusion.calibrator import LLRCalibrator, build_passport

if __name__ == "__main__":
    # Load existing metrics if available
    metrics_path = Path("reports/stage0/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
        logging.warning("No metrics.json found, building passport with empty metrics")

    cal = LLRCalibrator()
    path = build_passport(
        metrics=metrics,
        calibrator=cal,
        sprt_params={"sprt_alpha": 0.001, "sprt_beta": 0.05, "t_max_s": 30.0},
        model_path="data/models/passport.json",
    )
    print(f"Passport: {path}")
