from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from rris.data.text import contains_thai, normalize_text

logger = logging.getLogger(__name__)


EN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]", re.UNICODE)


@dataclass(frozen=True)
class MultilingualTokenizerConfig:
    keep_punct: bool = False
    thai_engine: str = "newmm"


class MultilingualTokenizer:
    """
    Tokenizer for baseline TF-IDF:
    - Thai: PyThaiNLP `word_tokenize`
    - English: regex tokenization (whitespace + punctuation preservation)
    """

    def __init__(self, cfg: MultilingualTokenizerConfig = MultilingualTokenizerConfig()) -> None:
        self.cfg = cfg
        try:
            from pythainlp.tokenize import word_tokenize as th_word_tokenize
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PyThaiNLP is required for Thai tokenization. Install `pythainlp`."
            ) from e
        self._th_word_tokenize = th_word_tokenize

    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        if not text:
            return []

        if contains_thai(text):
            toks = self._th_word_tokenize(
                text,
                engine=self.cfg.thai_engine,
                keep_whitespace=False,
            )
            toks = [t.strip() for t in toks if t and t.strip()]
        else:
            toks = EN_TOKEN_RE.findall(text)

        if not self.cfg.keep_punct:
            toks = [t for t in toks if re.search(r"[A-Za-z0-9\u0E00-\u0E7F]", t)]

        return toks

