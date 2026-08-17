"""Logger configuration for llm-utils-rag package."""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "llm_utils_rag",
    level: int = logging.ERROR,
    format_string: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and configure logger for llm-utils-rag package.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        log_dir: Directory for log files (default: ./logs)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    formatter = logging.Formatter(format_string)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create error file handler
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    error_log_file = log_path / f"{name}_error.log"

    # TimedRotatingFileHandler rotates at midnight and keeps 7 days
    error_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=error_log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    error_file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(error_file_handler)

    return logger


# Create default logger instance
logger = setup_logger()
