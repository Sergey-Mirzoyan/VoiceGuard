"""Test demo API and WebSocket with real models."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voiceguard.engine.api import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_file_analyze() -> None:
    client = TestClient(app)

    samples = [
        "data/samples/live_1.wav",
        "data/samples/live_2.wav",
        "data/samples/spoof_1.wav",
        "data/samples/spoof_2.wav",
    ]

    print("\n--- Проверка POST /v1/analyze ---")
    results = {}
    for sp in samples:
        p = Path(sp)
        if not p.exists():
            print(f"Missing: {sp}")
            continue
        with open(p, "rb") as f:
            resp = client.post("/v1/analyze", files={"file": (p.name, f, "audio/wav")})
        assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
        data = resp.json()
        results[p.name] = data
        scores = data.get("window_scores", [])
        n_segs = len(scores)
        s2_count = sum(1 for s in scores if s.get("s2") is not None)
        avg_s1 = np.mean([s["s1"] for s in scores if s.get("s1") is not None]) if scores else 0.0
        avg_s2 = np.mean([s["s2"] for s in scores if s.get("s2") is not None]) if s2_count else 0.0
        print(
            f"File: {p.name:<12} | Verdict: {data.get('verdict'):<12} | "
            f"Segs: {n_segs:<2} | s2 computed: {s2_count}/{n_segs} | "
            f"avg_s1: {avg_s1:+.2f} | avg_s2: {avg_s2:.2f}"
        )

    # Assertions per task requirements:
    # "нужны вердикты «Норма» на live и «Подозрение» хотя бы на одном spoof"
    live_verdicts = [
        results[f"live_{i}.wav"]["verdict"] for i in (1, 2) if f"live_{i}.wav" in results
    ]
    spoof_verdicts = [
        results[f"spoof_{i}.wav"]["verdict"] for i in (1, 2) if f"spoof_{i}.wav" in results
    ]

    print(f"Live verdicts: {live_verdicts}")
    print(f"Spoof verdicts: {spoof_verdicts}")

    assert "normal" in live_verdicts, f"Expected 'normal' for live, got {live_verdicts}"
    assert "suspicious" in spoof_verdicts, f"Expected 'suspicious' for spoof, got {spoof_verdicts}"
    print("✅ POST /v1/analyze PASSED!")


def test_websocket_stream() -> None:
    print("\n--- Проверка WebSocket /v1/stream ---")
    wav_path = Path("/tmp/speech_16k.wav")

    # Generate 5-second speech with say and afconvert
    try:
        aiff_path = Path("/tmp/speech.aiff")
        subprocess.run(
            [
                "say",
                "-v",
                "Milena",
                "Привет, это демонстрация потоковой защиты Voice Guard.",
                "-o",
                str(aiff_path),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)],
            check=True,
            capture_output=True,
        )
    except Exception as exc:
        print(f"say/afconvert fallback: {exc}")
        sr = 16000
        t = np.linspace(0, 5, sr * 5)
        sig = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
        sf.write(str(wav_path), sig.astype(np.float32), sr)

    samples, sr = sf.read(str(wav_path))
    dur_s = len(samples) / sr
    print(f"Audio for stream: {dur_s:.2f} s @ {sr} Hz ({len(samples)} samples)")

    pcm16 = (samples * 32767.0).astype(np.int16).tobytes()

    client = TestClient(app)
    received_messages: list[dict] = []
    stop_event = threading.Event()

    with client.websocket_connect("/v1/stream?channel=clean") as ws:
        def reader() -> None:
            while not stop_event.is_set():
                try:
                    msg = ws.receive_json(mode="text")
                    received_messages.append(msg)
                    print(
                        f"WS msg [{len(received_messages)}]: t={msg.get('t'):.1f}s | "
                        f"verdict={msg.get('verdict'):<10} | s1={msg.get('s1')} | "
                        f"s2={msg.get('s2')} | Lambda={msg.get('Lambda', 0.0):.2f}"
                    )
                except Exception:
                    break

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        # Send in 20 ms frames (320 samples * 2 bytes = 640 bytes)
        frame_bytes = 320 * 2
        for i in range(0, len(pcm16), frame_bytes):
            ws.send_bytes(pcm16[i : i + frame_bytes])
            time.sleep(0.015)  # simulate near-real-time streaming

        # Allow time for background D2 execution
        time.sleep(2.5)
        stop_event.set()
        ws.close()
        reader_thread.join(timeout=1.0)

    print(f"Total WS messages received: {len(received_messages)}")
    assert len(received_messages) >= 2, f"Expected >=2 msgs, got {len(received_messages)}"
    s2_present = any(m.get("s2") is not None for m in received_messages)
    print(f"s2 present in stream responses: {s2_present}")
    assert s2_present, "Expected s2 to appear in stream responses"
    print("✅ WebSocket /v1/stream PASSED!")


if __name__ == "__main__":
    test_file_analyze()
    test_websocket_stream()
