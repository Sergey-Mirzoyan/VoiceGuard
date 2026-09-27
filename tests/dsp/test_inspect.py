from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from voiceguard.dsp.__main__ import main as dsp_cli_main
from voiceguard.dsp.inspect import inspect_main, plot_inspection_pil
from voiceguard.dsp.vad import AudioClip


def test_ac07_inspect_creates_image_for_fixture(tmp_path: Path) -> None:
    """AC-07: inspect generates valid PNG inspection plot for a fixture clip."""
    out_img = tmp_path / "fixture_inspect.png"

    # Run inspect_main CLI
    exit_code = inspect_main(
        ["--clip-id", "fixture_test_01", "--channel", "clean", "--out", str(out_img)]
    )
    assert exit_code == 0
    assert out_img.is_file(), f"Expected {out_img} to exist"

    # Verify PNG header
    with open(out_img, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "File is not a valid PNG"

    # Verify with PIL
    with Image.open(out_img) as img:
        assert img.format == "PNG"
        assert img.size == (1000, 800)


def test_plot_inspection_pil_direct(tmp_path: Path) -> None:
    """Test plot_inspection_pil directly on synthetic AudioClip."""
    sr = 8000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    clip = AudioClip(samples=samples, sr=sr, clip_id="clip_direct")

    out_file = tmp_path / "direct.png"
    result = plot_inspection_pil(clip, channel="clean", out_path=out_file)

    assert result == out_file
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000


def test_dsp_main_cli_dispatch(tmp_path: Path) -> None:
    """Test python -m voiceguard.dsp inspect CLI dispatcher."""
    out_img = tmp_path / "cli_dispatch.png"
    code = dsp_cli_main(
        ["inspect", "--clip-id", "disp_test", "--channel", "clean", "--out", str(out_img)]
    )
    assert code == 0
    assert out_img.is_file()
