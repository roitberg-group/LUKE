from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console()


def get_logger(name: str = "luke", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=_console, show_time=False, show_path=False)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def set_global_log_level(level: str) -> None:
    for logger_name in ["luke", "luke.pipeline", "luke.ani", "luke.isolator"]:
        logging.getLogger(logger_name).setLevel(level.upper())
