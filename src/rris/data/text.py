from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")


def normalize_text(text: str) -> str:
    if text is None:
        raise ValueError("text is None")
    text = str(text)
    text = text.replace("\u200b", " ")  # zero-width space
    text = CTRL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_thai(text: str) -> bool:
    return bool(THAI_CHAR_RE.search(text or ""))


@dataclass(frozen=True)
class SentenceSplitConfig:
    # Keep this lightweight (works ok for EN/TH without heavy dependencies).
    max_len: int = 280


def split_sentences(text: str, cfg: SentenceSplitConfig = SentenceSplitConfig()) -> List[str]:
    """
    Simple multilingual sentence splitter.
    - For English: split on [.?!] boundaries.
    - For Thai: Thai doesn't use sentence punctuation consistently; we also split on newlines
      and keep max_len chunks.
    """

    text = normalize_text(text)
    if not text:
        return []

    # First split by hard separators/newlines.
    parts = re.split(r"(?:\r\n|\r|\n)+", text)
    parts = [p.strip() for p in parts if p.strip()]

    out: List[str] = []
    for p in parts:
        if contains_thai(p):
            # Thai fallback: chunk by max_len
            for i in range(0, len(p), cfg.max_len):
                chunk = p[i : i + cfg.max_len].strip()
                if chunk:
                    out.append(chunk)
        else:
            # English-ish splitting
            segs = re.split(r"(?<=[.!?])\s+", p)
            segs = [s.strip() for s in segs if s.strip()]
            out.extend(segs)

    return out

