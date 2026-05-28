"""
Progress helpers (tqdm wrappers, stage logging).

COMMON ERRORS:
  - ModuleNotFoundError: No module named 'tqdm' — pip install tqdm (in requirements).
  - Use --no_progress on CLI scripts when bars clutter logs or CI.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def tqdm_if(
    iterable: Iterable[T],
    *,
    show_progress: bool,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    **kwargs: Any,
) -> Iterable[T]:
    """Wrap *iterable* with tqdm when progress is enabled."""
    if not show_progress:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total, **kwargs)


def map_with_progress(
    fn: Callable[[T], Any],
    items: List[T],
    *,
    show_progress: bool,
    desc: str = "Processing",
    **tqdm_kwargs: Any,
) -> List[Any]:
    """Apply *fn* to each item, optionally showing a tqdm bar."""
    if not show_progress:
        return [fn(x) for x in items]
    from tqdm import tqdm

    return [fn(x) for x in tqdm(items, desc=desc, total=len(items), **tqdm_kwargs)]


@contextmanager
def log_stage(stage_name: str) -> Iterator[None]:
    """Log start/end of a pipeline stage."""
    logger.info("Stage start: %s", stage_name)
    try:
        yield
    finally:
        logger.info("Stage end: %s", stage_name)


@contextmanager
def stage(stage_name: str, *, enabled: bool = True) -> Iterator[None]:
    """Alias for log_stage (enabled kept for CLI compatibility)."""
    _ = enabled
    with log_stage(stage_name):
        yield
