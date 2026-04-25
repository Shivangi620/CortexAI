"""
infra/logger.py — Centralized structured logging for AutoML Studio.

Usage:
    from infra.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started", job_id=job_id, mode=mode)
"""
import logging
import sys
import json
from datetime import datetime, UTC
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Emit JSON log lines for easy parsing in production."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }

        for key, val in vars(record).items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName",
            ):
                try:
                    json.dumps({key: val})
                    log_obj[key] = val
                except Exception:
                    log_obj[key] = str(val)

        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


class HumanFormatter(logging.Formatter):
    """Plain, readable format for local development."""
    COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        base = f"{color}[{ts}] {record.levelname:<8}{self.RESET} {record.name} — {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def get_logger(name: str, structured: bool = False) -> logging.Logger:
    """
    Return a named logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    logger.addHandler(handler)

    import os
    from logging.handlers import RotatingFileHandler

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUNS_DIR = os.path.join(BASE_DIR, "runs")

    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(RUNS_DIR, "system.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=2
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter() if structured else HumanFormatter())
        logger.addHandler(file_handler)
    except Exception:
        pass

    logger.propagate = False
    return logger


_root_logger = get_logger("automl")
