from __future__ import annotations

import json
import logging
import os

import pytest

from voiceguard.logging import get_logger


def test_get_logger_standard(capsys: pytest.CaptureFixture[str]) -> None:
    # Ensure standard text format
    if "VG_LOG_JSON" in os.environ:
        del os.environ["VG_LOG_JSON"]

    logger = get_logger("test.standard")
    logger.info("Hello standard logging")

    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "test.standard" in captured.out
    assert "Hello standard logging" in captured.out


def test_get_logger_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VG_LOG_JSON", "1")
    # Reset handlers if any
    old_logger = logging.getLogger("test.json")
    old_logger.handlers.clear()

    logger = get_logger("test.json")
    logger.warning("Warning in JSON")

    captured = capsys.readouterr()
    assert captured.out.strip() != ""
    data = json.loads(captured.out.strip())
    assert data["level"] == "WARNING"
    assert data["logger"] == "test.json"
    assert data["message"] == "Warning in JSON"
    assert "timestamp" in data
