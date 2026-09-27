from __future__ import annotations

import shlex
import subprocess

import numpy as np
from scipy import signal

from voiceguard.channel.g711 import transcode_g711a, transcode_g711u
from voiceguard.channel.loss import apply_pcm_plc, generate_loss_mask
from voiceguard.channel.noise import add_noise
from voiceguard.config import Config, load_config
from voiceguard.types import AudioClip, ChannelSpec


def _transcode_amr_ffmpeg(
    samples: np.ndarray,
    sr: int,
    spec: ChannelSpec,
    ffmpeg_cmd_str: str,
) -> np.ndarray:
    """Encode and decode samples through AMR-NB or AMR-WB using ffmpeg subprocess pipes."""
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    ffmpeg_cmd = shlex.split(ffmpeg_cmd_str)

    if spec.codec == "amrnb":
        codec_name = "libopencore_amrnb"
    elif spec.codec == "amrwb":
        codec_name = "libvo_amrwbenc"
    else:
        raise ValueError(f"Unsupported AMR codec: {spec.codec}")

    bitrate_str = f"{spec.bitrate_kbps:g}k"

    enc_cmd = ffmpeg_cmd + [
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        codec_name,
        "-b:a",
        bitrate_str,
        "-f",
        "amr",
        "pipe:1",
    ]

    p_enc = subprocess.run(
        enc_cmd,
        input=pcm16.tobytes(),
        capture_output=True,
        check=False,
    )
    if p_enc.returncode != 0:
        err = p_enc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg AMR encoder failed (code {p_enc.returncode}): {err}")

    dec_cmd = ffmpeg_cmd + [
        "-loglevel",
        "error",
        "-f",
        "amr",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "pipe:1",
    ]

    p_dec = subprocess.run(
        dec_cmd,
        input=p_enc.stdout,
        capture_output=True,
        check=False,
    )
    if p_dec.returncode != 0:
        err = p_dec.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg AMR decoder failed (code {p_dec.returncode}): {err}")

    raw_pcm = np.frombuffer(p_dec.stdout, dtype=np.int16)
    return (raw_pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


def apply(
    clip: AudioClip,
    spec: ChannelSpec,
    seed: int,
    config: Config | None = None,
) -> AudioClip:
    """Apply telephone channel distortions according to spec."""
    if config is None:
        config = load_config()

    # Determine target sample rate
    if spec.codec in ("g711a", "g711u", "amrnb"):
        target_sr = 8000
    elif spec.codec == "amrwb":
        target_sr = 16000
    elif spec.codec == "clean":
        target_sr = clip.sr
    else:
        raise ValueError(f"Unknown codec: {spec.codec}")

    # 1. Resampling
    samples = clip.samples.astype(np.float32)
    if clip.sr != target_sr:
        gcd = int(np.gcd(clip.sr, target_sr))
        up = target_sr // gcd
        down = clip.sr // gcd
        samples = signal.resample_poly(samples, up, down).astype(np.float32)

    # 2. Additive Noise
    rng_noise = np.random.default_rng(seed)
    if spec.snr_db is not None:
        samples = add_noise(
            samples,
            sr=target_sr,
            snr_db=spec.snr_db,
            rng=rng_noise,
            noise_dir=config.channel.noise_dir,
        )

    # 3. Codec Encoding & Decoding
    if spec.codec == "clean":
        transcoded = samples.copy()
    elif spec.codec == "g711a":
        transcoded = transcode_g711a(samples)
    elif spec.codec == "g711u":
        transcoded = transcode_g711u(samples)
    elif spec.codec in ("amrnb", "amrwb"):
        transcoded = _transcode_amr_ffmpeg(
            samples,
            sr=target_sr,
            spec=spec,
            ffmpeg_cmd_str=config.channel.ffmpeg_cmd,
        )
    else:
        raise ValueError(f"Unsupported codec: {spec.codec}")

    # 4. Codec Delay Compensation & Exact Duration Matching
    if spec.codec in ("clean", "g711a", "g711u"):
        aligned = transcoded[: len(samples)]
        if len(aligned) < len(samples):
            aligned = np.pad(aligned, (0, len(samples) - len(aligned)))
    else:
        # Cross-correlation alignment
        ref_len = min(len(samples), target_sr)
        max_lag = int(0.1 * target_sr)  # search up to 100 ms
        if len(transcoded) >= ref_len and ref_len > 0:
            corr = signal.correlate(
                transcoded[: ref_len + max_lag],
                samples[:ref_len],
                mode="valid",
            )
            lag = int(np.argmax(corr))
        else:
            lag = 0

        aligned = transcoded[lag : lag + len(samples)]
        if len(aligned) < len(samples):
            aligned = np.pad(aligned, (0, len(samples) - len(aligned)))

    # 5. Packet Loss & PLC
    frame_len = int(target_sr * 0.02)
    n_frames = int(np.ceil(len(aligned) / frame_len))
    rng_loss = np.random.default_rng(seed + 1)

    loss_mask = generate_loss_mask(
        n_frames=n_frames,
        loss_pct=spec.loss_pct,
        rng=rng_loss,
        model=config.channel.loss_model,
        p=config.channel.gilbert_elliott.p,
        r=config.channel.gilbert_elliott.r,
    )
    if spec.loss_pct > 0.0:
        aligned = apply_pcm_plc(aligned, frame_len=frame_len, loss_mask=loss_mask)

    actual_loss_pct = float(np.mean(loss_mask) * 100.0) if n_frames > 0 else 0.0

    meta = dict(clip.meta)
    meta["channel"] = str(spec)
    meta["actual_loss_pct"] = actual_loss_pct
    meta["loss_pct"] = actual_loss_pct

    return AudioClip(
        samples=aligned.clip(-1.0, 1.0).astype(np.float32),
        sr=target_sr,
        clip_id=clip.clip_id,
        meta=meta,
    )
