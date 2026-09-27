from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from voiceguard.channel.cli import main, run_batch
from voiceguard.config import load_config


def test_channel_batch_cli(tmp_path: Path) -> None:
    # 1. Create a mini dummy manifest and audio files
    audio_dir = tmp_path / "raw"
    audio_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sr = 16000
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5)))).astype(
        np.float32
    )

    path1 = audio_dir / "clip_1.flac"
    path2 = audio_dir / "clip_2.flac"
    sf.write(path1, sine, sr)
    sf.write(path2, sine, sr)

    manifest_df = pd.DataFrame(
        [
            {"clip_id": "clip_1", "path": str(path1), "split": "test"},
            {"clip_id": "clip_2", "path": str(path2), "split": "test"},
        ]
    )
    manifest_path = data_dir / "manifest.parquet"
    manifest_df.to_parquet(manifest_path)

    # 2. Run batch processing
    cfg = load_config(overrides={"paths": {"data": str(data_dir)}})
    ret = run_batch(
        split="test",
        channels=["g711a"],
        workers=2,
        manifest_path=manifest_path,
        config=cfg,
    )
    assert ret == 0

    # 3. Verify output files exist
    out_dir = data_dir / "channel" / "g711a"
    assert (out_dir / "clip_1.flac").is_file()
    assert (out_dir / "clip_2.flac").is_file()

    # 4. Check cache hit on rerun
    mtime1 = (out_dir / "clip_1.flac").stat().st_mtime
    ret2 = run_batch(
        split="test",
        channels=["g711a"],
        workers=2,
        manifest_path=manifest_path,
        config=cfg,
    )
    assert ret2 == 0
    mtime2 = (out_dir / "clip_1.flac").stat().st_mtime
    assert mtime1 == mtime2


def test_cli_main_missing_manifest(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "non_existent.parquet"
    ret = main(["run", "--channels", "g711a", "--manifest", str(missing_manifest)])
    assert ret == 1
