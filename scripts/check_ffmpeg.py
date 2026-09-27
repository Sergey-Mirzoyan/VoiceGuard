#!/usr/bin/env python3
"""Check that ffmpeg is installed and has required AMR-NB/AMR-WB encoders and decoders."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("ERROR: ffmpeg binary not found in PATH", file=sys.stderr)
        return 1

    try:
        encoders_proc = subprocess.run(
            [ffmpeg_bin, "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        encoders_output = encoders_proc.stdout
    except Exception as e:
        print(f"ERROR: Failed to run ffmpeg -encoders: {e}", file=sys.stderr)
        return 1

    try:
        decoders_proc = subprocess.run(
            [ffmpeg_bin, "-decoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        decoders_output = decoders_proc.stdout
    except Exception as e:
        print(f"ERROR: Failed to run ffmpeg -decoders: {e}", file=sys.stderr)
        return 1

    required_encoders = ["libopencore_amrnb", "libvo_amrwbenc"]
    required_decoders = ["amrnb", "amrwb"]

    missing: list[str] = []

    print("Checking ffmpeg AMR encoders and decoders:")
    for enc in required_encoders:
        found = any(
            enc == token or enc in line.split()
            for line in encoders_output.splitlines()
            for token in line.split()
        )
        if found:
            print(f"  [OK] Encoder found: {enc}")
        else:
            print(f"  [FAIL] Missing encoder: {enc}")
            missing.append(f"encoder {enc}")

    for dec in required_decoders:
        found = any(
            dec == token or dec in line.split()
            for line in decoders_output.splitlines()
            for token in line.split()
        )
        if found:
            print(f"  [OK] Decoder found: {dec}")
        else:
            print(f"  [FAIL] Missing decoder: {dec}")
            missing.append(f"decoder {dec}")

    if missing:
        print(
            f"\nERROR: Missing required components: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    print("\nAll required AMR encoders and decoders are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
