"""FastAPI application (упрощённое ЧТЗ-09).

Endpoints:
  POST /v1/analyze        — file upload analysis
  WS   /v1/stream         — real-time PCM16 streaming
  GET  /v1/passport       — detector passport
  GET  /v1/health         — health check
  GET  /                  — serve UI (index.html)
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceGuard AM",
    description="Synthetic speech detection API",
    version="0.1.0",
)

# Static files
_UI_DIR = Path(__file__).parent.parent / "ui" / "static"
_MODELS_DIR = Path("data/models")
_PASSPORT_PATH = _MODELS_DIR / "passport.json"

# Cache the engine per channel
_engines: dict[str, Any] = {}


def _get_engine(channel: str = "clean") -> Any:
    from voiceguard.config import load_config
    from voiceguard.engine.stream_engine import StreamEngine

    if channel not in _engines:
        cfg = load_config()
        _engines[channel] = StreamEngine(cfg=cfg, channel=channel, models_dir=_MODELS_DIR)
    return _engines[channel]


def _load_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Load audio from bytes, return (samples_float32, sr)."""
    import soundfile as sf

    try:
        buf = io.BytesIO(data)
        samples, sr = sf.read(buf)
        samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return samples, sr
    except Exception:
        pass

    # m4a/aac/webm/opus etc.: decode with ffmpeg to 16 kHz mono float32
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(
            status_code=400,
            detail="Формат не поддерживается без ffmpeg (brew install ffmpeg). "
            "Без него работают WAV, MP3, OGG, FLAC.",
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", "pipe:0"]
        + ["-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode(errors="replace").strip()[-300:]
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 16000


def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(16000, sr)
        return np.asarray(resample_poly(samples, 16000 // g, sr // g), dtype=np.float32)
    except Exception:
        import librosa

        return np.asarray(
            librosa.resample(samples, orig_sr=sr, target_sr=16000), dtype=np.float32
        )


# ---------------------------------------------------------------------------
# Serve UI
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def ui_root() -> FileResponse:
    index = _UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index))


# Mount static files if dir exists
if _UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health() -> JSONResponse:
    d1_ok = (_MODELS_DIR / "d1_nb.joblib").exists()
    return JSONResponse(
        {
            "status": "ok",
            "d1_model": d1_ok,
            "passport": _PASSPORT_PATH.exists(),
        }
    )


# ---------------------------------------------------------------------------
# GET /v1/passport
# ---------------------------------------------------------------------------


@app.get("/v1/passport")
async def passport() -> JSONResponse:
    if not _PASSPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Passport not found. Run stage0 first.")
    with open(_PASSPORT_PATH) as f:
        data = json.load(f)
    return JSONResponse(data)


# ---------------------------------------------------------------------------
# POST /v1/analyze
# ---------------------------------------------------------------------------


@app.post("/v1/analyze")
async def analyze(
    file: UploadFile = File(...),  # noqa: B008
    channel: str = Form(default="clean"),  # noqa: B008
) -> JSONResponse:
    """Analyze an uploaded audio file.

    Args:
        file: Audio file (wav, mp3, m4a, ogg, webm, max 60 s).
        channel: Channel spec string (clean | g711a | amrnb_12.2).

    Returns:
        JSON with window_scores list, verdict, d2_details.
    """
    data = await file.read()
    if len(data) > 60 * 256 * 1024:  # rough 60s limit at ~256kbps
        raise HTTPException(status_code=413, detail="File too large (max 60 s audio)")

    samples, sr = _load_audio(data, file.filename or "")
    if sr != 16000:
        samples = _resample_to_16k(samples, sr)
        sr = 16000

    # Limit duration to 60 s
    max_samples = 60 * sr
    samples = samples[:max_samples]

    engine = _get_engine(channel)
    window_scores = engine.process_audio(samples, sr, channel_override=channel)

    final_verdict = "undecided"
    for ws in reversed(window_scores):
        if ws.verdict != "pending":
            final_verdict = ws.verdict
            break

    return JSONResponse(
        {
            "channel": channel,
            "duration_s": len(samples) / sr,
            "n_segments": len(window_scores),
            "verdict": final_verdict,
            "window_scores": [
                {
                    "t": ws.t,
                    "s1": ws.s1,
                    "s2": ws.s2,
                    "llr": ws.llr,
                    "Lambda": ws.Lambda,
                    "verdict": ws.verdict,
                }
                for ws in window_scores
            ],
        }
    )


def _make_vad(aggressiveness: int) -> Any:
    """Create webrtcvad.Vad, or None if unavailable (then every frame is speech)."""
    import sys
    import types

    # webrtcvad imports pkg_resources only for its version; newer setuptools drop it
    if "pkg_resources" not in sys.modules:
        try:
            import pkg_resources  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            dummy = types.ModuleType("pkg_resources")
            dummy.get_distribution = lambda n: types.SimpleNamespace(version="2.0.10")  # type: ignore[attr-defined]
            sys.modules["pkg_resources"] = dummy
    try:
        import webrtcvad

        return webrtcvad.Vad(aggressiveness)
    except Exception as exc:
        logger.warning("webrtcvad unavailable (%s), treating all frames as speech", exc)
        return None


# ---------------------------------------------------------------------------
# WS /v1/stream
# ---------------------------------------------------------------------------


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket, channel: str = "clean") -> None:
    """WebSocket streaming endpoint.

    Client sends raw PCM16 mono 16 kHz frames.
    Server responds every 1 second of speech with a JSON WindowScore.
    """
    await websocket.accept()

    from voiceguard.config import load_config
    from voiceguard.engine.stream_engine import StreamEngine
    from voiceguard.types import Segment

    cfg = load_config()
    engine = StreamEngine(cfg=cfg, channel=channel, models_dir=_MODELS_DIR)

    SR = 16000
    FRAME_MS = 20
    FRAME_LEN = SR * FRAME_MS // 1000  # 320 samples
    SEG_FRAMES = 50  # 1 s = 50 * 20 ms frames

    pending = np.zeros(0, dtype=np.float32)  # samples not yet forming a full frame
    speech_frames: list[np.ndarray] = []
    t_elapsed = 0.0
    vad = _make_vad(cfg.vad.aggressiveness)

    try:
        while True:
            raw = await websocket.receive_bytes()
            if len(raw) < 2:
                continue

            # Parse PCM16 samples; browsers send small chunks (e.g. 128 samples),
            # so carry the remainder over to the next message.
            pcm16 = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
            pending = np.concatenate([pending, pcm16.astype(np.float32) / 32768.0])

            # Process in frame chunks
            offset = 0
            while offset + FRAME_LEN <= len(pending):
                frame = pending[offset : offset + FRAME_LEN]
                offset += FRAME_LEN

                # VAD check
                if vad is None:
                    is_speech = True
                else:
                    frame_pcm16 = (frame * 32767).astype(np.int16)
                    try:
                        is_speech = vad.is_speech(frame_pcm16.tobytes(), SR)
                    except Exception:
                        is_speech = True  # assume speech on error

                if is_speech:
                    speech_frames.append(frame)

                if len(speech_frames) >= SEG_FRAMES:
                    # Build segment
                    seg_samples = np.concatenate(speech_frames[:SEG_FRAMES]).astype(np.float32)
                    speech_frames = speech_frames[SEG_FRAMES:]

                    voiced_mask = np.ones(SEG_FRAMES, dtype=bool)
                    seg = Segment(
                        samples=seg_samples,
                        sr=SR,
                        voiced_mask=voiced_mask,
                        clip_id="stream",
                        index=int(t_elapsed),
                    )

                    ws_result = engine.process_segment(seg)
                    t_elapsed += 1.0

                    await websocket.send_json(
                        {
                            "t": ws_result.t,
                            "s1": ws_result.s1,
                            "s2": ws_result.s2,
                            "llr": ws_result.llr,
                            "Lambda": ws_result.Lambda,
                            "verdict": ws_result.verdict,
                        }
                    )

            pending = pending[offset:]

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
