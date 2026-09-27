"""Data loader for ЧТЗ-02 (упрощённое).

Downloads Hy-Generated dataset via datasets streaming, takes N clips per class,
splits by speaker into calib/train/test (30/40/30), normalises to 16 kHz FLAC,
builds data/manifest.parquet.
"""
from __future__ import annotations

import hashlib
import io
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

logger = logging.getLogger(__name__)

# Dataset parameters
DATASET_NAME = "ErikMkrtchyan/Hy-Generated-audio-data-with-cv20.0"
# The dataset has no label column: the class is given by the split.
# train/test/eval are Common Voice 20.0 (live), generated is F5-TTS (spoof).
LIVE_SPLITS = ("train", "test", "eval")
SPOOF_SPLIT = "generated"
# Common Voice rows are grouped by speaker; cap clips per speaker so the
# speaker-level calib/train/test split has enough distinct speakers.
MAX_CLIPS_PER_SPEAKER = 5
CLIPS_PER_CLASS = 300
TARGET_SR = 16000
SPLIT_CALIB = 0.30
SPLIT_TRAIN = 0.40
# SPLIT_TEST = 0.30 (remainder)


def _resample_to_16k(samples: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 16 kHz using scipy if needed."""
    if orig_sr == TARGET_SR:
        return samples.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(TARGET_SR, orig_sr)
        up = TARGET_SR // g
        down = orig_sr // g
        resampled = resample_poly(samples.astype(np.float64), up, down)
        return np.asarray(resampled, dtype=np.float32)
    except Exception as exc:
        logger.warning("Resample failed (%s), using librosa", exc)
        import librosa

        return np.asarray(
            librosa.resample(samples.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR),
            dtype=np.float32,
        )


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert to mono by averaging channels."""
    if samples.ndim == 1:
        return samples
    return np.asarray(samples.mean(axis=1), dtype=np.float32)


def _speaker_id_from_meta(row: dict[str, Any]) -> str:
    """Extract or derive speaker ID from dataset row metadata."""
    for key in ("speaker_id", "client_id", "speaker", "spk_id", "spk"):
        val = row.get(key)
        if val is not None:
            return str(val)
    # Fallback: hash of audio path/id
    for key in ("audio_id", "id", "file", "path"):
        val = row.get(key)
        if val is not None:
            return hashlib.md5(str(val).encode()).hexdigest()[:8]
    return "unknown"


def _split_speakers(
    speakers: list[str],
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split speakers into calib/train/test sets (30/40/30)."""
    rng = random.Random(seed)
    s = sorted(set(speakers))
    rng.shuffle(s)
    n = len(s)
    n_calib = max(1, int(n * SPLIT_CALIB))
    n_train = max(1, int(n * SPLIT_TRAIN))
    calib = s[:n_calib]
    train = s[n_calib : n_calib + n_train]
    test = s[n_calib + n_train :]
    if not test:
        test = [s[-1]]
        train = s[n_calib : n_calib + n_train - 1] or s[n_calib:-1]
    return calib, train, test


def build_manifest(
    out_dir: str | Path = "data",
    n_per_class: int = CLIPS_PER_CLASS,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Download Hy-Generated, normalise audio, build manifest.parquet.

    Args:
        out_dir: Directory to write audio files and manifest.
        n_per_class: Number of clips to collect per class (live/spoof).
        seed: Random seed.
        force: Overwrite existing manifest if True.

    Returns:
        Path to manifest.parquet
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.parquet"

    if manifest_path.exists() and not force:
        logger.info("Manifest already exists at %s, skipping download", manifest_path)
        return manifest_path

    audio_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install 'datasets' package: pip install datasets") from e

    logger.info("Loading %s via streaming...", DATASET_NAME)

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {"live": 0, "spoof": 0}
    per_speaker: dict[str, int] = {}

    def _labelled_rows() -> Any:
        """Interleave live and spoof rows so both classes fill up together."""
        live = (
            row
            for split in LIVE_SPLITS
            for row in load_dataset(DATASET_NAME, split=split, streaming=True)
        )
        spoof = iter(load_dataset(DATASET_NAME, split=SPOOF_SPLIT, streaming=True))
        sources = [("live", live), ("spoof", spoof)]
        while sources:
            for item in list(sources):
                label, it = item
                try:
                    yield label, next(it)
                except StopIteration:
                    sources.remove(item)

    for label, row in _labelled_rows():
        if counts["live"] >= n_per_class and counts["spoof"] >= n_per_class:
            break

        if counts[label] >= n_per_class:
            continue

        spk = _speaker_id_from_meta(row)
        if per_speaker.get(spk, 0) >= MAX_CLIPS_PER_SPEAKER:
            continue

        # Extract audio
        audio_data = row.get("audio")
        if audio_data is None:
            continue

        try:
            if isinstance(audio_data, dict):
                samples = np.array(audio_data["array"], dtype=np.float32)
                sr_orig = int(audio_data["sampling_rate"])
            elif isinstance(audio_data, bytes):
                buf = io.BytesIO(audio_data)
                samples, sr_orig = sf.read(buf)
                samples = samples.astype(np.float32)
            else:
                continue
        except Exception as exc:
            logger.debug("Failed to read audio: %s", exc)
            continue

        samples = _to_mono(samples)
        if sr_orig != TARGET_SR:
            samples = _resample_to_16k(samples, sr_orig)

        # Clip normalization
        peak = np.abs(samples).max()
        if peak > 0:
            samples = samples / peak * 0.9

        clip_id = f"{label}_{counts[label]:04d}"
        flac_path = audio_dir / f"{clip_id}.flac"

        try:
            sf.write(str(flac_path), samples, TARGET_SR, format="FLAC")
        except Exception as exc:
            logger.debug("Failed to write FLAC: %s", exc)
            continue

        rows.append(
            {
                "clip_id": clip_id,
                "label": label,
                "speaker_id": spk,
                "path": str(flac_path.relative_to(out_dir)),
                "duration_s": len(samples) / TARGET_SR,
                "sr": TARGET_SR,
                "split": "",  # will be filled below
            }
        )
        counts[label] += 1
        per_speaker[spk] = per_speaker.get(spk, 0) + 1
        if counts[label] % 50 == 0:
            logger.info("Collected %d %s clips", counts[label], label)

    logger.info("Collected: live=%d, spoof=%d", counts["live"], counts["spoof"])

    if not rows:
        raise RuntimeError("No audio rows collected from dataset")

    df = pd.DataFrame(rows)

    # Split by speakers
    all_spk = df["speaker_id"].tolist()
    calib_spk, train_spk, test_spk = _split_speakers(all_spk, seed=seed)
    calib_set = set(calib_spk)
    train_set = set(train_spk)
    test_set = set(test_spk)

    def assign_split(spk: str) -> str:
        if spk in calib_set:
            return "calib"
        if spk in train_set:
            return "train"
        if spk in test_set:
            return "test"
        # Fallback: assign proportionally
        rng = random.Random(hash(spk) + seed)
        r = rng.random()
        if r < SPLIT_CALIB:
            return "calib"
        if r < SPLIT_CALIB + SPLIT_TRAIN:
            return "train"
        return "test"

    df["split"] = df["speaker_id"].apply(assign_split)

    # If all speakers are "unknown", fall back to index-based split
    if (df["split"] == "").all() or df["speaker_id"].nunique() == 1:
        rng = random.Random(seed)
        indices = list(range(len(df)))
        rng.shuffle(indices)
        n = len(indices)
        n_c = int(n * SPLIT_CALIB)
        n_tr = int(n * SPLIT_TRAIN)
        splits = [""] * n
        for i in indices[:n_c]:
            splits[i] = "calib"
        for i in indices[n_c : n_c + n_tr]:
            splits[i] = "train"
        for i in indices[n_c + n_tr :]:
            splits[i] = "test"
        df["split"] = splits

    # Write parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, manifest_path)
    logger.info("Manifest written to %s (%d rows)", manifest_path, len(df))

    # Summary
    logger.info("Split counts:\n%s", df.groupby(["split", "label"]).size().to_string())
    return manifest_path


def load_clips(
    manifest_path: str | Path = "data/manifest.parquet",
    split: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Load manifest with optional filtering by split and/or label."""
    df = pd.read_parquet(manifest_path)
    if split is not None:
        df = df[df["split"] == split]
    if label is not None:
        df = df[df["label"] == label]
    return df.reset_index(drop=True)
