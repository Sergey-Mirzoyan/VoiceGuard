from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "docs/legacy")
import analysis as orig_module

from tests.prng_fixtures import generate_lfsr, generate_mt, generate_urandom
from voiceguard.config import load_config
from voiceguard.d2 import legacy
from voiceguard.d2.predictors import dissertation_verdict

warnings.filterwarnings("ignore")

CHECK_KEYS = [
    "bit_w8_next",
    "bit_w8_prev",
    "bit_w16_next",
    "bit_w16_prev",
    "bit_w32_next",
    "bit_w32_prev",
    "block_w8_next",
    "block_w8_prev",
    "block_w16_next",
    "block_w16_prev",
    "block_w32_next",
    "block_w32_prev",
]

DISS_KEY_MAP = {
    "bit_w8_next": "bit_w8_fwd",
    "bit_w8_prev": "bit_w8_bwd",
    "bit_w16_next": "bit_w16_fwd",
    "bit_w16_prev": "bit_w16_bwd",
    "bit_w32_next": "bit_w32_fwd",
    "bit_w32_prev": "bit_w32_bwd",
    "block_w8_next": "block_w8_fwd",
    "block_w8_prev": "block_w8_bwd",
    "block_w16_next": "block_w16_fwd",
    "block_w16_prev": "block_w16_bwd",
    "block_w32_next": "block_w32_fwd",
    "block_w32_prev": "block_w32_bwd",
}


def run_orig(bits: list[int], seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    res = orig_module.run_full_analysis(bits, window_sizes=[8, 16, 32])
    if not res:
        return {}
    deltas = {}
    for d in res["bitwise"]["neural_network"]["details"]:
        key = f"bit_w{d['window_size']}_{d['direction']}"
        deltas[key] = float(d["accuracy"] - 0.5)

    block_accs: dict[str, list[float]] = {}
    for d in res["block"]["neural_network"]["details"]:
        key = f"block_w{d['window_size']}_{d['direction']}"
        block_accs.setdefault(key, []).append(d["accuracy"])

    for k, v in block_accs.items():
        deltas[k] = float(np.mean(v) - 0.5)

    delta_max = max(deltas.values())
    is_strong = (
        res["bitwise"]["neural_network"]["is_strong"]
        and res["block"]["neural_network"]["is_strong"]
    )
    verdict = "соответствие ТСБ" if is_strong else "нарушение ТСБ"
    threshold = 0.05

    return {
        "deltas": deltas,
        "delta_max": delta_max,
        "threshold": threshold,
        "verdict": verdict,
        "is_violation": not is_strong,
    }


def run_legacy(bits: list[int], seed: int) -> dict[str, Any]:
    res = legacy.run_full_analysis(bits, window_sizes=[8, 16, 32], seed=seed)
    if not res:
        return {}
    deltas = {}
    for d in res["bitwise"]["neural_network"]["details"]:
        key = f"bit_w{d['window_size']}_{d['direction']}"
        deltas[key] = float(d["accuracy"] - 0.5)

    block_accs: dict[str, list[float]] = {}
    for d in res["block"]["neural_network"]["details"]:
        key = f"block_w{d['window_size']}_{d['direction']}"
        block_accs.setdefault(key, []).append(d["accuracy"])

    for k, v in block_accs.items():
        deltas[k] = float(np.mean(v) - 0.5)

    delta_max = max(deltas.values())
    is_strong = (
        res["bitwise"]["neural_network"]["is_strong"]
        and res["block"]["neural_network"]["is_strong"]
    )
    verdict = "соответствие ТСБ" if is_strong else "нарушение ТСБ"
    threshold = 0.05

    return {
        "deltas": deltas,
        "delta_max": delta_max,
        "threshold": threshold,
        "verdict": verdict,
        "is_violation": not is_strong,
    }


def run_dissertation(bits_arr: np.ndarray, seed: int) -> dict[str, Any]:
    cfg = load_config(overrides={"d2": {"seed": seed}})
    res = dissertation_verdict(bits_arr, cfg=cfg)
    return {
        "deltas": res["deltas"],
        "delta_max": res["delta_max"],
        "threshold": res["eps"],
        "verdict": res["verdict"],
        "is_violation": res["is_violation"],
    }


def process_single(gen_name: str, seed: int) -> dict[str, Any]:
    if gen_name == "lfsr":
        bits_arr = generate_lfsr(100000, seed=seed)
    elif gen_name == "mt":
        bits_arr = generate_mt(100000, seed=seed)
    else:
        # os.urandom
        bits_arr = generate_urandom(100000)

    bits_list = [int(b) for b in bits_arr]

    r_orig = run_orig(bits_list, seed)
    r_leg = run_legacy(bits_list, seed)
    r_diss = run_dissertation(bits_arr, seed)

    # Check exact match between orig and legacy
    for k in r_orig["deltas"]:
        diff = abs(r_orig["deltas"][k] - r_leg["deltas"][k])
        assert diff < 1e-9, (
            f"Mismatch in {gen_name} s={seed} {k}: {r_orig['deltas'][k]} != {r_leg['deltas'][k]}"
        )

    return {
        "gen": gen_name,
        "seed": seed,
        "orig": r_orig,
        "legacy": r_leg,
        "diss": r_diss,
    }


def format_table(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(
        "# Сводная таблица сравнения docs/legacy/analysis.py, legacy.py и dissertation_verdict\n"
    )
    lines.append(
        "Параметры: n = 100 000 бит, 10 семян на каждый генератор (РСЛОС, Вихрь Мерсенна, os.urandom).\n"  # noqa: E501
    )
    lines.append("Порог analysis.py и legacy.py: $\\delta_{пор} = 0.05$ (accuracy $\\le 0.55$).\n")
    lines.append(
        "Порог dissertation_verdict: $\\varepsilon = \\frac{z_{\\alpha/(2m)}}{2\\sqrt{N_{test}}} \\approx 0.0118$ (при $\\alpha=0.01, m=12, N_{test}=20000$).\n\n"  # noqa: E501
    )

    # Group results by generator
    gens = ["lfsr", "mt", "urandom"]
    gen_titles = {
        "lfsr": "1. РСЛОС (LFSR, 32-бит максимальной длины)",
        "mt": "2. Вихрь Мерсенна (Mersenne Twister / random.getrandbits)",
        "urandom": "3. Криптографический генератор ОС (os.urandom)",
    }

    for g in gens:
        lines.append(f"### {gen_titles[g]}\n")
        lines.append(
            "| S | Модуль | b8_f | b8_b | b16_f | b16_b | b32_f | b32_b | blk8_f | blk8_b | bl16_f | bl16_b | bl32_f | bl32_b | δ_общ | Пор. | Вердикт |"  # noqa: E501
        )
        lines.append(
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|"  # noqa: E501
        )

        g_results = [r for r in results if r["gen"] == g]
        for item in g_results:
            s = item["seed"]
            # Row 1: original
            r_orig = item["orig"]
            r_leg = item["legacy"]
            r_diss = item["diss"]

            def fmt_deltas(d_dict: dict[str, float], is_diss: bool = False) -> list[str]:
                res = []
                for k in CHECK_KEYS:
                    lookup_k = DISS_KEY_MAP[k] if is_diss else k
                    v = d_dict.get(lookup_k, 0.0)
                    res.append(f"{v:+.4f}")
                return res

            orig_d = fmt_deltas(r_orig["deltas"])
            leg_d = fmt_deltas(r_leg["deltas"])
            diss_d = fmt_deltas(r_diss["deltas"], is_diss=True)

            lines.append(
                f"| {s} | **analysis.py** | "
                + " | ".join(orig_d)
                + f" | **{r_orig['delta_max']:+.4f}** | {r_orig['threshold']:.4f} | {r_orig['verdict']} |"  # noqa: E501
            )
            lines.append(
                f"| {s} | **legacy.py** | "
                + " | ".join(leg_d)
                + f" | **{r_leg['delta_max']:+.4f}** | {r_leg['threshold']:.4f} | {r_leg['verdict']} |"  # noqa: E501
            )
            lines.append(
                f"| {s} | **dissertation** | "
                + " | ".join(diss_d)
                + f" | **{r_diss['delta_max']:+.4f}** | {r_diss['threshold']:.4f} | {r_diss['verdict']} |"  # noqa: E501
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        lines.append("\n")

    return "\n".join(lines)


if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    gens = ["lfsr", "mt", "urandom"]

    tasks = [(g, s) for g in gens for s in seeds]
    print(f"Running comparison for {len(tasks)} runs in parallel (n_jobs=4)...")
    results = joblib.Parallel(n_jobs=4, verbose=10)(
        joblib.delayed(process_single)(g, s) for g, s in tasks
    )

    import pickle

    os.makedirs("reports", exist_ok=True)
    with open("reports/comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved comparison_results.pkl successfully!")

    md_report = format_table(results)
    with open("reports/comparison_report.md", "w") as f:
        f.write(md_report)
    print("Saved reports/comparison_report.md successfully!")
