from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from PIL import Image, ImageDraw

from voiceguard.channel.simulator import apply
from voiceguard.config import Config, load_config
from voiceguard.dsp.lpc import residual
from voiceguard.dsp.vad import compute_vad_mask, compute_voiced_mask
from voiceguard.types import AudioClip, ChannelSpec


def generate_synthetic_fixture(sr: int = 16000, duration_s: float = 3.0) -> np.ndarray:
    """Generate a synthetic speech-like vowel signal with silence gaps for testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # 0.5s speech, 0.5s silence, 1.5s speech, 0.5s silence
    gate = ((t >= 0.2) & (t <= 1.0)) | ((t >= 1.5) & (t <= 2.8))

    f0 = 140.0
    pulse = np.zeros_like(t)
    pulse[:: int(sr / f0)] = 1.0

    from scipy import signal

    b, a = signal.iirpeak(500.0, 500.0 / 80.0, fs=sr)
    vowel = signal.lfilter(b, a, pulse) * gate
    norm_vowel = vowel / (np.max(np.abs(vowel)) + 1e-6) * 0.75
    result: np.ndarray = np.asarray(norm_vowel, dtype=np.float32)
    return result


def plot_inspection_pil(
    clip: AudioClip,
    channel: str,
    out_path: Path,
    cfg: Config | None = None,
) -> Path:
    """Draw 4-panel inspection plot (waveform, residual, VAD, pitch) and save PNG."""
    if cfg is None:
        cfg = load_config()

    spec = ChannelSpec.parse(channel)
    processed = apply(clip, spec, seed=42, config=cfg)

    samples = processed.samples
    sr = processed.sr
    duration = len(samples) / float(sr)

    # 1. Residual
    res = residual(samples, sr, cfg=cfg)

    # 2. VAD & Voicing masks
    frame_ms = cfg.audio.frame_ms
    vad_mask = compute_vad_mask(
        samples, sr=sr, frame_ms=frame_ms, aggressiveness=cfg.vad.aggressiveness
    )
    voiced_mask = compute_voiced_mask(samples, sr=sr, frame_ms=frame_ms, vad_mask=vad_mask)

    # 3. Pitch contour
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(sr))
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)

    # Dimensions
    w, h = 1000, 800
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_left, margin_right = 70, 40
    panel_w = w - margin_left - margin_right
    panel_h = 140
    gaps = 40

    def draw_panel_border(y_top: int, title: str) -> None:
        draw.rectangle(
            [margin_left, y_top, margin_left + panel_w, y_top + panel_h], outline=(200, 200, 200)
        )
        draw.text((margin_left, y_top - 20), title, fill=(30, 30, 30))

    # Panel 1: Waveform
    y1 = 40
    draw_panel_border(
        y1, f"1. Processed Waveform (Channel: {channel}, sr={sr} Hz, {duration:.2f} s)"
    )
    if len(samples) > 0:
        step = max(1, len(samples) // panel_w)
        downsampled = samples[::step][:panel_w]
        mid_y = y1 + panel_h // 2
        pts = [
            (margin_left + i, int(mid_y - downsampled[i] * (panel_h // 2 - 5)))
            for i in range(len(downsampled))
        ]
        if len(pts) > 1:
            draw.line(pts, fill=(31, 119, 180), width=1)

    # Panel 2: LPC Residual
    y2 = y1 + panel_h + gaps
    draw_panel_border(y2, "2. LPC Residual e[n]")
    if len(res) > 0:
        max_r = float(np.max(np.abs(res))) or 1.0
        step = max(1, len(res) // panel_w)
        downsampled_r = res[::step][:panel_w] / max_r
        mid_y = y2 + panel_h // 2
        pts_r = [
            (margin_left + i, int(mid_y - downsampled_r[i] * (panel_h // 2 - 5)))
            for i in range(len(downsampled_r))
        ]
        if len(pts_r) > 1:
            draw.line(pts_r, fill=(214, 39, 40), width=1)

    # Panel 3: VAD and Voicing Activity
    y3 = y2 + panel_h + gaps
    draw_panel_border(y3, "3. VAD (Blue) and Voiced Speech (Green)")
    n_frames = len(vad_mask)
    if n_frames > 0:
        block_w = max(1.0, panel_w / float(n_frames))
        for i in range(n_frames):
            bx0 = int(margin_left + i * block_w)
            bx1 = int(margin_left + (i + 1) * block_w)
            if voiced_mask[i]:
                color = (44, 160, 44)  # Green: voiced speech
            elif vad_mask[i]:
                color = (31, 119, 180)  # Blue: unvoiced speech
            else:
                color = (230, 230, 230)  # Gray: silence
            draw.rectangle([bx0, y3 + 20, bx1, y3 + panel_h - 20], fill=color)

    # Panel 4: Pitch Contour
    y4 = y3 + panel_h + gaps
    draw_panel_border(y4, "4. Pitch Contour F0 (75 - 600 Hz)")
    n_pts_pitch = 200
    pitch_times = np.linspace(0.0, duration, n_pts_pitch)
    pitch_vals = [float(pitch.get_value_at_time(t)) for t in pitch_times]
    pitch_pts = []
    for i, val in enumerate(pitch_vals):
        if not np.isnan(val) and val > 0:
            norm_val = np.clip((val - 75.0) / (600.0 - 75.0), 0.0, 1.0)
            py = int(y4 + panel_h - 10 - norm_val * (panel_h - 20))
            px = int(margin_left + i * (panel_w / float(n_pts_pitch)))
            pitch_pts.append((px, py))

    for px, py in pitch_pts:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(148, 103, 189))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def inspect_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for voiceguard.dsp inspect."""
    parser = argparse.ArgumentParser(description="VoiceGuard DSP Inspection Tool")
    parser.add_argument("--clip-id", type=str, required=True, help="Clip identifier")
    parser.add_argument(
        "--channel", type=str, default="clean", help="Channel specification (e.g. amrnb_12.2)"
    )
    parser.add_argument(
        "--audio", type=str, default=None, help="Path to input audio file (optional)"
    )
    parser.add_argument("--out", type=str, default=None, help="Output image file path (optional)")

    args = parser.parse_args(argv)

    cfg = load_config()

    if args.audio and Path(args.audio).is_file():
        samples, sr = sf.read(args.audio, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        clip = AudioClip(samples=samples, sr=sr, clip_id=args.clip_id)
    else:
        # Check standard data locations or fallback to synthetic fixture
        candidate = Path(cfg.paths.data) / "processed" / f"{args.clip_id}.flac"
        if candidate.is_file():
            samples, sr = sf.read(candidate, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            clip = AudioClip(samples=samples, sr=sr, clip_id=args.clip_id)
        else:
            # Generate synthetic test clip for this clip_id
            samples = generate_synthetic_fixture(sr=16000, duration_s=3.0)
            clip = AudioClip(samples=samples, sr=16000, clip_id=args.clip_id)

    out_p = (
        Path(args.out) if args.out else Path(cfg.paths.reports) / "inspect" / f"{args.clip_id}.png"
    )
    result_path = plot_inspection_pil(clip, channel=args.channel, out_path=out_p, cfg=cfg)
    print(f"Inspection plot saved to: {result_path}")
    return 0
