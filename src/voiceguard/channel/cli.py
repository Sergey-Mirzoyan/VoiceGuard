from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf

from voiceguard.channel.simulator import apply
from voiceguard.config import Config, load_config
from voiceguard.logging import get_logger
from voiceguard.types import AudioClip, ChannelSpec

logger = get_logger("voiceguard.channel.cli")


def _process_single_clip(
    clip_id: str,
    src_path_str: str,
    channel_str: str,
    out_dir_str: str,
    seed: int,
    config: Config,
) -> bool:
    """Process a single clip through a channel if not already cached."""
    out_path = Path(out_dir_str) / f"{clip_id}.flac"
    if out_path.is_file():
        return True  # Cache hit

    src_path = Path(src_path_str)
    if not src_path.is_file():
        logger.warning("Source file not found: %s", src_path)
        return False

    spec = ChannelSpec.parse(channel_str)
    samples, sr = sf.read(src_path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    clip = AudioClip(samples=samples, sr=sr, clip_id=clip_id)
    processed = apply(clip, spec, seed=seed, config=config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, processed.samples, processed.sr)
    return True


def run_batch(
    split: str,
    channels: list[str],
    workers: int = 4,
    manifest_path: Path | None = None,
    config: Config | None = None,
) -> int:
    """Batch process manifest clips through channel simulator with parallel caching."""
    if config is None:
        config = load_config()

    if manifest_path is None:
        manifest_path = Path(config.paths.data) / "manifest.parquet"

    if not manifest_path.is_file():
        print(f"ERROR: Manifest file not found at: {manifest_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(manifest_path)
    if "split" in df.columns and split != "all":
        df = df[df["split"] == split]

    if len(df) == 0:
        print(f"No clips found for split: {split}")
        return 0

    print(f"Processing {len(df)} clips across {len(channels)} channels with {workers} workers...")

    tasks = []
    base_data_dir = Path(config.paths.data)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for ch_str in channels:
            # Validate channel spec
            spec = ChannelSpec.parse(ch_str)
            norm_ch_str = str(spec)
            ch_out_dir = base_data_dir / "channel" / norm_ch_str
            ch_out_dir.mkdir(parents=True, exist_ok=True)

            for idx, row in df.iterrows():
                clip_id = str(row["clip_id"])
                src_path = str(row["path"])
                task_seed = int(config.d2.seed) + int(idx)
                tasks.append(
                    executor.submit(
                        _process_single_clip,
                        clip_id,
                        src_path,
                        norm_ch_str,
                        str(ch_out_dir),
                        task_seed,
                        config,
                    )
                )

        completed = 0
        total = len(tasks)
        for fut in as_completed(tasks):
            fut.result()
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} clips processed")

    print(f"Batch processing completed successfully: {completed} tasks.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for voiceguard.channel."""
    parser = argparse.ArgumentParser(description="VoiceGuard channel simulator batch processing")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run batch processing on manifest clips")
    run_parser.add_argument(
        "--split", type=str, default="test", help="Dataset split (e.g. test, train)"
    )
    run_parser.add_argument(
        "--channels",
        type=str,
        required=True,
        help="Comma-separated channel specifications (e.g. amrnb_12.2,g711a)",
    )
    run_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    run_parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.parquet")

    args = parser.parse_args(argv)

    if args.command == "run":
        channel_list = [ch.strip() for ch in args.channels.split(",") if ch.strip()]
        manifest_p = Path(args.manifest) if args.manifest else None
        return run_batch(
            split=args.split,
            channels=channel_list,
            workers=args.workers,
            manifest_path=manifest_p,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
