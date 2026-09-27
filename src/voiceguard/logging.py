from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


class Iso8601Formatter(logging.Formatter):
    """Standard formatter with ISO-8601 timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        return dt.isoformat()

    def format(self, record: logging.LogRecord) -> str:
        record.asctime = self.formatTime(record)
        return f"{record.asctime} [{record.levelname}] {record.name}: {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        log_entry: dict[str, Any] = {
            "timestamp": dt.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Get or create a configured logger."""
    logger = logging.getLogger(name)

    level_name = os.environ.get("VG_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger or root is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        use_json = os.environ.get("VG_LOG_JSON", "0").lower() in ("1", "true", "yes")
        if use_json:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(Iso8601Formatter())

        logger.addHandler(handler)
        logger.propagate = False

    return logger
