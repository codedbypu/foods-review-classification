from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json_like: bool = False


def setup_logging(cfg: LoggingConfig = LoggingConfig()) -> None:
    """
    Set up process-wide logging.
    - Keeps format stable for CLI + pipelines
    - Avoids duplicate handlers on repeated imports
    """

    root = logging.getLogger()
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root.setLevel(level)

    if root.handlers:
        # Respect existing handlers (e.g., in notebooks), but update level.
        for h in root.handlers:
            h.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if cfg.json_like:
        fmt = (
            '{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s",'
            '"msg":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)

