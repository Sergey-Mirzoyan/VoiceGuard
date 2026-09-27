"""CLI entry point for data module: build manifest from Hy-Generated."""
from __future__ import annotations

import logging

from voiceguard.data.loader import build_manifest

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = build_manifest(out_dir="data", n_per_class=300, seed=42)
    print(f"Manifest: {path}")
