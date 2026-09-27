from __future__ import annotations

import shlex
import subprocess
from typing import Any

import numpy as np

from voiceguard.channel.g711 import transcode_g711a, transcode_g711u
from voiceguard.channel.loss import apply_pcm_plc, generate_loss_mask
from voiceguard.channel.noise import add_noise
from voiceguard.config import Config, load_config
from voiceguard.types import ChannelSpec

_WORKER_SCRIPT = """
import sys
import ctypes
import numpy as np

codec = sys.argv[1]
bitrate = float(sys.argv[2])
loss_pct = float(sys.argv[3])
seed = int(sys.argv[4])

rng = np.random.default_rng(seed)

if codec == "amrnb":
    frame_len = 160
    frame_bytes = 320
    lib = ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libopencore-amrnb.so.0')
    lib.Encoder_Interface_init.restype = ctypes.c_void_p
    lib.Encoder_Interface_init.argtypes = [ctypes.c_int]
    lib.Encoder_Interface_Encode.restype = ctypes.c_int
    lib.Encoder_Interface_Encode.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_short),
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
    ]
    lib.Decoder_Interface_init.restype = ctypes.c_void_p
    lib.Decoder_Interface_Decode.restype = None
    lib.Decoder_Interface_Decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int
    ]
    lib.Encoder_Interface_exit.restype = None
    lib.Encoder_Interface_exit.argtypes = [ctypes.c_void_p]
    lib.Decoder_Interface_exit.restype = None
    lib.Decoder_Interface_exit.argtypes = [ctypes.c_void_p]

    nb_modes = {4.75: 0, 5.15: 1, 5.9: 2, 6.7: 3, 7.4: 4, 7.95: 5, 10.2: 6, 12.2: 7}
    mode = nb_modes.get(bitrate, 7)
    enc = lib.Encoder_Interface_init(0)
    dec = lib.Decoder_Interface_init()

    out_bytes = (ctypes.c_ubyte * 32)()
    dec_speech = (ctypes.c_short * 160)()

    try:
        while True:
            raw = sys.stdin.buffer.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            speech = np.frombuffer(raw, dtype=np.int16)
            speech_ptr = speech.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
            lib.Encoder_Interface_Encode(enc, mode, speech_ptr, out_bytes, 0)

            is_lost = bool(loss_pct > 0.0 and rng.uniform(0.0, 100.0) < loss_pct)
            bfi = 1 if is_lost else 0
            lib.Decoder_Interface_Decode(dec, out_bytes, dec_speech, bfi)
            sys.stdout.buffer.write(bytes(dec_speech))
            sys.stdout.buffer.flush()
    finally:
        lib.Encoder_Interface_exit(enc)
        lib.Decoder_Interface_exit(dec)

elif codec == "amrwb":
    frame_len = 320
    frame_bytes = 640
    lib_enc = ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libvo-amrwbenc.so.0')
    lib_enc.E_IF_init.restype = ctypes.c_void_p
    lib_enc.E_IF_encode.restype = ctypes.c_int
    lib_enc.E_IF_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_short),
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
    ]
    lib_enc.E_IF_exit.restype = None
    lib_enc.E_IF_exit.argtypes = [ctypes.c_void_p]

    lib_dec = ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libopencore-amrwb.so.0')
    lib_dec.D_IF_init.restype = ctypes.c_void_p
    lib_dec.D_IF_decode.restype = None
    lib_dec.D_IF_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_short),
        ctypes.c_int
    ]
    lib_dec.D_IF_exit.restype = None
    lib_dec.D_IF_exit.argtypes = [ctypes.c_void_p]

    wb_modes = {
        6.6: 0, 8.85: 1, 12.65: 2, 14.25: 3, 15.85: 4, 18.25: 5, 19.85: 6, 23.05: 7, 23.85: 8
    }
    mode = wb_modes.get(bitrate, 2)

    enc = lib_enc.E_IF_init()
    dec = lib_dec.D_IF_init()
    out_bytes = (ctypes.c_ubyte * 64)()
    dec_speech = (ctypes.c_short * 320)()

    try:
        while True:
            raw = sys.stdin.buffer.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            speech = np.frombuffer(raw, dtype=np.int16)
            speech_ptr = speech.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
            lib_enc.E_IF_encode(enc, mode, speech_ptr, out_bytes, 0)

            is_lost = bool(loss_pct > 0.0 and rng.uniform(0.0, 100.0) < loss_pct)
            bfi = 1 if is_lost else 0
            lib_dec.D_IF_decode(dec, out_bytes, dec_speech, bfi)
            sys.stdout.buffer.write(bytes(dec_speech))
            sys.stdout.buffer.flush()
    finally:
        lib_enc.E_IF_exit(enc)
        lib_dec.D_IF_exit(dec)
"""


class StreamTranscoder:
    """Streaming channel transcoder for 20 ms frames with <= 60 ms delay."""

    def __init__(
        self,
        spec: ChannelSpec,
        seed: int = 42,
        config: Config | None = None,
    ) -> None:
        self.spec = spec
        self.seed = seed
        self.config = config or load_config()

        if spec.codec in ("g711a", "g711u", "amrnb"):
            self.sr = 8000
        elif spec.codec == "amrwb":
            self.sr = 16000
        elif spec.codec == "clean":
            self.sr = 8000  # default for clean in stream
        else:
            raise ValueError(f"Unknown codec: {spec.codec}")

        self.frame_len = int(self.sr * 0.02)
        self.rng_noise = np.random.default_rng(seed)
        self.rng_loss = np.random.default_rng(seed + 1)
        self._proc: subprocess.Popen[bytes] | None = None

        if self.spec.codec in ("amrnb", "amrwb"):
            self._start_worker()

    def _start_worker(self) -> None:
        cmd_parts = shlex.split(self.config.channel.ffmpeg_cmd)
        # Check if docker invocation
        if "docker" in cmd_parts and "run" in cmd_parts:
            # Extract docker prefix before ffmpeg
            try:
                ff_idx = cmd_parts.index("ffmpeg")
                docker_prefix = cmd_parts[:ff_idx]
            except ValueError:
                docker_prefix = ["docker", "run", "--rm", "-i", "voiceguard-base:latest"]
            worker_cmd = docker_prefix + [
                "python3",
                "-u",
                "-c",
                _WORKER_SCRIPT,
                self.spec.codec,
                str(self.spec.bitrate_kbps or 12.2),
                str(self.spec.loss_pct),
                str(self.seed),
            ]
        else:
            # Local python execution
            worker_cmd = [
                "python3",
                "-u",
                "-c",
                _WORKER_SCRIPT,
                self.spec.codec,
                str(self.spec.bitrate_kbps or 12.2),
                str(self.spec.loss_pct),
                str(self.seed),
            ]

        self._proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def push(self, pcm: np.ndarray) -> np.ndarray:
        """Process one 20 ms PCM frame with delay <= 60 ms."""
        # Ensure float32 array
        samples = pcm.astype(np.float32)

        # 1. Noise
        if self.spec.snr_db is not None:
            samples = add_noise(
                samples,
                sr=self.sr,
                snr_db=self.spec.snr_db,
                rng=self.rng_noise,
                noise_dir=self.config.channel.noise_dir,
            )

        # 2. Transcode
        if self.spec.codec == "clean":
            out = samples.copy()
            if self.spec.loss_pct > 0.0:
                mask = generate_loss_mask(1, self.spec.loss_pct, self.rng_loss)
                out = apply_pcm_plc(out, self.frame_len, mask)
            return out

        if self.spec.codec == "g711a":
            out = transcode_g711a(samples)
            if self.spec.loss_pct > 0.0:
                mask = generate_loss_mask(1, self.spec.loss_pct, self.rng_loss)
                out = apply_pcm_plc(out, self.frame_len, mask)
            return out

        if self.spec.codec == "g711u":
            out = transcode_g711u(samples)
            if self.spec.loss_pct > 0.0:
                mask = generate_loss_mask(1, self.spec.loss_pct, self.rng_loss)
                out = apply_pcm_plc(out, self.frame_len, mask)
            return out

        # AMR worker stream
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Transcoder worker is not running")

        pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
        expected_bytes = self.frame_len * 2
        raw_bytes = pcm16.tobytes()

        self._proc.stdin.write(raw_bytes)
        self._proc.stdin.flush()

        res_bytes = self._proc.stdout.read(expected_bytes)
        if len(res_bytes) < expected_bytes:
            raise RuntimeError(
                f"Stream transcoder read error: expected {expected_bytes} bytes, "
                f"got {len(res_bytes)}"
            )

        res_pcm = np.frombuffer(res_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        return res_pcm.clip(-1.0, 1.0)

    def close(self) -> None:
        """Close the underlying worker process."""
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
            finally:
                self._proc = None

    def __enter__(self) -> StreamTranscoder:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
