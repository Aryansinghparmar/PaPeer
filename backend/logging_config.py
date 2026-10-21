"""Small logging helper with a safe default format."""

from __future__ import annotations

import logging

from backend.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
