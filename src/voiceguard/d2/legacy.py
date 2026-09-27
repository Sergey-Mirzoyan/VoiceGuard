from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import chi2, chi2_contingency
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def read_binary_string(content: str) -> list[int] | None:
    """Parses binary string content."""
    try:
        sequence = content.strip()
        if not all(bit in "01" for bit in sequence):
            return None
        return [int(bit) for bit in sequence]
    except Exception:
        return None


def analyze_sequence_direction(
    sequence: list[int] | np.ndarray,
    window_size: int,
    start_pos: int,
    direction: str = "next",
    seed: int = 42,
    return_details: bool = True,
) -> tuple[list[dict[str, Any]], float, int]:
    """Analysis of sequence in one direction using Neural Network (exact port of analysis.py)."""
    seq = list(sequence)
    X_train: list[list[int]] = []
    y_train: list[int] = []

    # Define training size as 20% of available data, max 50
    training_size = min(len(seq) // 5, 50)

    # Form training set around start_pos
    if direction == "next":
        train_range = range(
            max(0, start_pos - training_size // 2),
            min(start_pos + training_size // 2, len(seq) - window_size),
        )
        for i in train_range:
            X_train.append(seq[i : i + window_size])
            y_train.append(seq[i + window_size])

        test_range = range(start_pos, len(seq) - window_size)
    else:
        train_range = range(
            min(len(seq) - 1, start_pos + training_size // 2),
            max(window_size - 1, start_pos - training_size // 2),
            -1,
        )
        for i in train_range:
            X_train.append(seq[i - window_size : i])
            y_train.append(seq[i - window_size - 1])

        test_range = range(start_pos, window_size - 1, -1)

    if len(X_train) == 0:
        return [], 0.0, 0

    # Create and train model with exact legacy parameters
    clf = MLPClassifier(hidden_layer_sizes=(window_size,), max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)

    if not test_range:
        return [], 0.0, 0

    # Prepare test inputs
    if direction == "next":
        X_test = [seq[i : i + window_size] for i in test_range]
        y_test = [seq[i + window_size] for i in test_range]
        positions = [i + window_size for i in test_range]
    else:
        X_test = [seq[i - window_size : i] for i in test_range]
        y_test = [seq[i - window_size - 1] for i in test_range]
        positions = [i - window_size - 1 for i in test_range]

    # Predict in batch (mathematically identical to single predict per window, but 1000x faster)
    preds = clf.predict(X_test)
    matches = preds == y_test
    correct_predictions = int(np.sum(matches))
    accuracy = correct_predictions / len(y_test) if y_test else 0.0

    predictions: list[dict[str, Any]] = []
    if return_details:
        for pos, w, pr, act, corr in zip(positions, X_test, preds, y_test, matches, strict=True):
            predictions.append(
                {
                    "position": pos,
                    "window": "".join(map(str, w)),
                    "predicted": int(pr),
                    "actual": int(act),
                    "correct": bool(corr),
                }
            )

    return predictions, float(accuracy), len(y_test)


def statistical_prediction(
    sequence: list[int] | np.ndarray,
    window_size: int,
    start_pos: int,
    direction: str = "next",
) -> tuple[list[dict[str, Any]], float, float, int]:
    """Statistical analysis using Chi-Square method (exact port of analysis.py)."""
    seq = list(sequence)
    predictions: list[dict[str, Any]] = []
    total_chi_square = 0.0
    total_p_value = 0.0
    total_tests = 0

    training_size = min(len(seq) // 5, 50)

    if direction == "next":
        train_range = range(
            max(0, start_pos - training_size // 2),
            min(start_pos + training_size // 2, len(seq) - window_size),
        )
        test_range = range(start_pos, len(seq) - window_size)
    else:
        train_range = range(
            min(len(seq) - 1, start_pos + training_size // 2),
            max(window_size - 1, start_pos - training_size // 2),
            -1,
        )
        test_range = range(start_pos, window_size - 1, -1)

    window_stats: dict[tuple[int, ...], Counter] = {}
    for i in train_range:
        if direction == "next":
            window = tuple(seq[i : i + window_size])
            next_bit = seq[i + window_size]
        else:
            window = tuple(seq[i - window_size : i])
            next_bit = seq[i - window_size - 1]

        if window not in window_stats:
            window_stats[window] = Counter()
        window_stats[window][next_bit] += 1

    for i in test_range:
        if direction == "next":
            window = tuple(seq[i : i + window_size])
            actual = seq[i + window_size]
            pos = i + window_size
        else:
            window = tuple(seq[i - window_size : i])
            actual = seq[i - window_size - 1]
            pos = i - window_size - 1

        freq = window_stats.get(window, Counter())

        if sum(freq.values()) > 0:
            expected = sum(freq.values()) / 2
            chi_square = sum((obs - expected) ** 2 / expected for obs in freq.values())
            p_value = 1.0 - chi2.cdf(chi_square, df=1)

            predictions.append(
                {
                    "position": pos,
                    "window": "".join(map(str, window)),
                    "actual": int(actual),
                    "chi_square": float(chi_square),
                    "p_value": float(p_value),
                }
            )
            total_chi_square += chi_square
            total_p_value += p_value
            total_tests += 1

    avg_chi_square = total_chi_square / total_tests if total_tests > 0 else 0.0
    avg_p_value = total_p_value / total_tests if total_tests > 0 else 0.0
    return predictions, avg_chi_square, avg_p_value, total_tests


def get_cached_block_model(
    sequence: list[int] | np.ndarray,
    window_size: int,
    seed: int = 42,
) -> tuple[MLPRegressor | None, StandardScaler | None]:
    """Train cached block model (exact port of analysis.py)."""
    seq = list(sequence)
    valid_indices = list(range(len(seq) - window_size))
    if len(valid_indices) > 50:
        indices = random.sample(valid_indices, 50)
    else:
        indices = valid_indices

    X_train: list[list[int]] = []
    y_train: list[int] = []
    for i in indices:
        X_train.append(seq[i : i + window_size])
        y_train.append(seq[i + window_size])

    if not X_train:
        return None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(window_size * 2, window_size),
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def block_predict_with_cached_model(
    sequence: list[int] | np.ndarray,
    position: int,
    window_size: int,
    predict_size: int,
    direction: str,
    model: MLPRegressor | None,
    scaler: StandardScaler | None,
) -> dict[str, Any] | None:
    """Autoregressive block prediction using cached model (exact port of analysis.py)."""
    if model is None or scaler is None:
        return None

    seq = list(sequence)
    if direction == "next":
        if position + window_size + predict_size > len(seq):
            return None
        window = seq[position : position + window_size]
        actual = seq[position + window_size : position + window_size + predict_size]
    else:
        if position - window_size - predict_size < 0:
            return None
        window = seq[position - window_size : position]
        actual = seq[position - window_size - predict_size : position - window_size]

    predicted: list[int] = []
    current = np.array(window)

    for _ in range(predict_size):
        current_scaled = scaler.transform(current.reshape(1, -1))
        pred = model.predict(current_scaled)[0]
        bit = 1 if pred >= 0.5 else 0
        predicted.append(bit)

        if direction == "next":
            current = np.roll(current, -1)
            current[-1] = bit
        else:
            current = np.roll(current, 1)
            current[0] = bit

    matches = sum(a == b for a, b in zip(predicted, actual, strict=True))
    accuracy = matches / predict_size if predict_size > 0 else 0.0

    return {
        "window": "".join(map(str, window)),
        "predicted": "".join(map(str, predicted)),
        "actual": "".join(map(str, actual)),
        "accuracy": float(accuracy),
    }


def get_block_transitions(
    sequence: list[int] | np.ndarray,
    window_size: int,
    predict_size: int,
) -> dict[tuple[int, ...], dict[tuple[int, ...], int]]:
    """Build block transition counts table."""
    seq = list(sequence)
    transitions: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
    for i in range(len(seq) - window_size - predict_size):
        current = tuple(seq[i : i + window_size])
        next_block = tuple(seq[i + window_size : i + window_size + predict_size])
        if current not in transitions:
            transitions[current] = {}
        if next_block not in transitions[current]:
            transitions[current][next_block] = 0
        transitions[current][next_block] += 1
    return transitions


def block_statistical_prediction_cached(
    sequence: list[int] | np.ndarray,
    position: int,
    window_size: int,
    predict_size: int,
    direction: str,
    transitions: dict[tuple[int, ...], dict[tuple[int, ...], int]],
) -> dict[str, Any] | None:
    """Cached statistical block prediction (exact port of analysis.py)."""
    seq = list(sequence)
    if direction == "next":
        if position + window_size + predict_size > len(seq):
            return None
        window = tuple(seq[position : position + window_size])
        actual = tuple(seq[position + window_size : position + window_size + predict_size])
    else:
        if position - window_size - predict_size < 0:
            return None
        window = tuple(seq[position - window_size : position])
        actual = tuple(seq[position - window_size - predict_size : position - window_size])

    if window not in transitions:
        return {
            "window": "".join(map(str, window)),
            "predicted": "0" * predict_size,
            "actual": "".join(map(str, actual)),
            "chi_square": 0.0,
            "p_value": 1.0,
        }

    observed = np.zeros((2, 2))
    for next_block, count in transitions[window].items():
        row = 0 if next_block == actual else 1
        observed[row][0] += count
        observed[row][1] += sum(transitions[window].values()) - count

    chi2_val, p_value, _, _ = chi2_contingency(observed + 1)
    predicted = max(transitions[window].items(), key=lambda x: x[1])[0]

    return {
        "window": "".join(map(str, window)),
        "predicted": "".join(map(str, predicted)),
        "actual": "".join(map(str, actual)),
        "chi_square": float(chi2_val),
        "p_value": float(p_value),
    }


def run_full_analysis(
    sequence: list[int] | np.ndarray,
    window_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Runs full analysis on the sequence (exact port of analysis.py)."""
    if window_sizes is None:
        window_sizes = [8, 16, 32]
    if len(sequence) == 0:
        return None

    random.seed(seed)
    np.random.seed(seed)

    seq = list(sequence)
    start_pos = len(seq) // 2

    results: dict[str, Any] = {
        "sequence_length": len(seq),
        "bitwise": {
            "neural_network": {
                "details": [],
                "overall_accuracy": 0.0,
                "total_analyzed": 0,
                "is_strong": False,
            },
            "statistical": {
                "details": [],
                "overall_chi_square": 0.0,
                "overall_p_value": 0.0,
                "total_tests": 0,
                "is_strong": False,
            },
        },
        "block": {
            "neural_network": {
                "details": [],
                "overall_accuracy": 0.0,
                "total_analyzed": 0,
                "is_strong": False,
            },
            "statistical": {
                "details": [],
                "overall_chi_square": 0.0,
                "overall_p_value": 0.0,
                "total_tests": 0,
                "is_strong": False,
            },
        },
    }

    # === BITWISE ANALYSIS ===
    total_acc = 0.0
    total_an = 0
    for ws in window_sizes:
        for d in ["next", "prev"]:
            preds, acc, bits = analyze_sequence_direction(
                seq, ws, start_pos, d, seed=seed, return_details=False
            )
            if bits > 0:
                results["bitwise"]["neural_network"]["details"].append(
                    {
                        "window_size": ws,
                        "direction": d,
                        "bits_analyzed": bits,
                        "accuracy": acc,
                        "predictions": preds,
                    }
                )
                total_acc += acc * bits
                total_an += bits
    if total_an > 0:
        results["bitwise"]["neural_network"]["overall_accuracy"] = total_acc / total_an
        results["bitwise"]["neural_network"]["total_analyzed"] = total_an
        results["bitwise"]["neural_network"]["is_strong"] = (total_acc / total_an) <= 0.55

    tot_chi = 0.0
    tot_p = 0.0
    tot_t = 0
    for ws in window_sizes:
        for d in ["next", "prev"]:
            preds, avg_chi, avg_p, tests = statistical_prediction(seq, ws, start_pos, d)
            if tests > 0:
                results["bitwise"]["statistical"]["details"].append(
                    {
                        "window_size": ws,
                        "direction": d,
                        "tests": tests,
                        "avg_chi_square": avg_chi,
                        "avg_p_value": avg_p,
                        "predictions": preds,
                    }
                )
                tot_chi += avg_chi * tests
                tot_p += avg_p * tests
                tot_t += tests
    if tot_t > 0:
        results["bitwise"]["statistical"]["overall_chi_square"] = tot_chi / tot_t
        results["bitwise"]["statistical"]["overall_p_value"] = tot_p / tot_t
        results["bitwise"]["statistical"]["total_tests"] = tot_t
        results["bitwise"]["statistical"]["is_strong"] = (tot_p / tot_t) > 0.05

    # === BLOCKWISE ANALYSIS ===
    min_pos = 32
    max_pos = len(seq) - 32
    if max_pos > min_pos:
        positions = [random.randint(min_pos, max_pos) for _ in range(50)]

        block_nn_acc = 0.0
        block_nn_bits = 0

        block_stat_p = 0.0
        block_stat_count = 0

        for ws in window_sizes:
            model, scaler = get_cached_block_model(seq, ws, seed=seed)
            transitions = get_block_transitions(seq, ws, ws)

            for pos in positions:
                for d in ["next", "prev"]:
                    res_nn = block_predict_with_cached_model(seq, pos, ws, ws, d, model, scaler)
                    if res_nn:
                        results["block"]["neural_network"]["details"].append(
                            {
                                "window_size": ws,
                                "direction": d,
                                "window": res_nn["window"],
                                "predicted": res_nn["predicted"],
                                "actual": res_nn["actual"],
                                "accuracy": res_nn["accuracy"],
                            }
                        )
                        block_nn_acc += res_nn["accuracy"] * ws
                        block_nn_bits += ws

                    res_stat = block_statistical_prediction_cached(seq, pos, ws, ws, d, transitions)
                    if res_stat:
                        results["block"]["statistical"]["details"].append(
                            {
                                "window_size": ws,
                                "direction": d,
                                "window": res_stat["window"],
                                "predicted": res_stat["predicted"],
                                "actual": res_stat["actual"],
                                "chi_square": res_stat["chi_square"],
                                "p_value": res_stat["p_value"],
                            }
                        )
                        block_stat_p += res_stat["p_value"]
                        block_stat_count += 1

        if block_nn_bits > 0:
            results["block"]["neural_network"]["overall_accuracy"] = block_nn_acc / block_nn_bits
            results["block"]["neural_network"]["total_analyzed"] = block_nn_bits
            results["block"]["neural_network"]["is_strong"] = (block_nn_acc / block_nn_bits) <= 0.55

        if block_stat_count > 0:
            results["block"]["statistical"]["overall_p_value"] = block_stat_p / block_stat_count
            results["block"]["statistical"]["total_tests"] = block_stat_count
            results["block"]["statistical"]["is_strong"] = (block_stat_p / block_stat_count) > 0.05

    return results


def run_legacy_checks(
    sequence: list[int] | np.ndarray,
    window_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all 12 NN checks using legacy logic and return structured deltas and verdicts."""
    res = run_full_analysis(sequence, window_sizes=window_sizes, seed=seed)
    if not res:
        return {}

    deltas: dict[str, float] = {}
    for d in res["bitwise"]["neural_network"]["details"]:
        d_name = "fwd" if d["direction"] == "next" else "bwd"
        cid = f"bit_w{d['window_size']}_{d_name}"
        deltas[cid] = float(d["accuracy"] - 0.5)

    block_accs: dict[str, list[float]] = {}
    for d in res["block"]["neural_network"]["details"]:
        d_name = "fwd" if d["direction"] == "next" else "bwd"
        cid = f"block_w{d['window_size']}_{d_name}"
        block_accs.setdefault(cid, []).append(d["accuracy"])

    for cid, vals in block_accs.items():
        deltas[cid] = float(np.mean(vals) - 0.5)

    delta_max = float(max(deltas.values())) if deltas else 0.0
    is_strong = (
        res["bitwise"]["neural_network"]["is_strong"]
        and res["block"]["neural_network"]["is_strong"]
    )
    verdict_legacy = "соответствие ТСБ" if is_strong else "нарушение ТСБ"

    # Dissertation threshold
    z = float(stats.norm.isf(0.01 / 24.0))
    n_test = len(sequence) // 2  # 50,000 for n=100,000
    eps = float(z / (2.0 * np.sqrt(n_test)))
    verdict_dissertation = "нарушение ТСБ" if delta_max > eps else "соответствие ТСБ"

    return {
        "deltas": deltas,
        "delta_max": delta_max,
        "threshold_legacy": 0.05,
        "threshold_dissertation": eps,
        "verdict_legacy": verdict_legacy,
        "verdict_dissertation": verdict_dissertation,
        "is_strong": is_strong,
        "bitwise_acc": res["bitwise"]["neural_network"]["overall_accuracy"],
        "block_acc": res["block"]["neural_network"]["overall_accuracy"],
    }
