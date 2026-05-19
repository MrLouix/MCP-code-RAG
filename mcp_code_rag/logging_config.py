"""Logging configuration for the application."""

import json
import logging
from pathlib import Path


def setup_logging(level: str = "INFO", format: str = "text", file: str | None = None) -> None:
    """
    Configure the root logger with specified level, format, and optional file handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format type ("text" or "json").
        file: Optional path to log file. If None, logs only to console.

    Raises:
        ValueError: If level or format are invalid.
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}")

    valid_formats = ["text", "json"]
    if format not in valid_formats:
        raise ValueError(f"format must be one of {valid_formats}")

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    if format == "json":
        formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if file:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class _JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Create named logger for the module
logger = logging.getLogger("mcp_code_rag")
