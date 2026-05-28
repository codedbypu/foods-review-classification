"""
Extract aspect mentions per review sentence (keyword/heuristic).

COMMON ERRORS:
  - Kernel crash / disk full: embedding fallback loads SentenceTransformer per surface without cache.
    Fix: enable_embeddings_fallback=False in notebooks; use --skip_aspects in score_and_flag.
  - Called in a per-review loop — keep mapping lightweight (see aspects.py @lru_cache).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rris.data.aspects import ASPECT_CANONICAL, AspectMappingConfig, aspect_keyword_map, map_surface_to_aspect
from rris.data.text import SentenceSplitConfig, contains_thai, normalize_text, split_sentences

logger = logging.getLogger(__name__)


_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")


@dataclass(frozen=True)
class AspectExtractionConfig:
    sentence_split: SentenceSplitConfig = SentenceSplitConfig()
    mapping: AspectMappingConfig = AspectMappingConfig()
    min_surface_len: int = 2
    max_surfaces_per_sentence: int = 6


@dataclass(frozen=True)
class AspectMention:
    review_id: str
    sentence_idx: int
    sentence_text: str
    surface: str
    aspect: str


def _extract_candidate_surfaces_en(sentence: str, kw_map: Dict[str, str]) -> List[str]:
    s = sentence.lower()
    surfaces: List[str] = []
    # Prefer keyword hits (including multiword keys) by substring search.
    for kw in kw_map.keys():
        if not kw:
            continue
        if kw in s:
            surfaces.append(kw)
    if surfaces:
        # Deduplicate while preserving order
        seen = set()
        out = []
        for x in surfaces:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Fallback: pick top "noun-like" words by regex, then map via fuzzy/embeddings.
    words = _EN_WORD_RE.findall(sentence)
    # Keep unique short list
    seen = set()
    for w in words:
        lw = w.lower()
        if lw not in seen:
            surfaces.append(lw)
            seen.add(lw)
        if len(surfaces) >= 12:
            break
    return surfaces


def _extract_candidate_surfaces_th(sentence: str, kw_map: Dict[str, str]) -> List[str]:
    s = sentence
    surfaces: List[str] = []

    # Try high-precision keyword substring matches first.
    for kw in kw_map.keys():
        if not kw:
            continue
        # Only consider Thai keywords for Thai sentences to reduce noise
        if contains_thai(kw) and kw in s:
            surfaces.append(kw)

    if surfaces:
        seen = set()
        out = []
        for x in surfaces:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Optional POS-aware fallback (best-effort).
    try:
        from pythainlp import word_tokenize
        from pythainlp.tag import pos_tag
    except Exception:
        return surfaces

    toks = [t for t in word_tokenize(sentence, engine="newmm", keep_whitespace=False) if t.strip()]
    try:
        tagged = pos_tag(toks, corpus="orchid_ud")
    except Exception:
        tagged = [(t, "") for t in toks]

    # Collect noun-ish tokens (corpus tags vary; keep broad)
    for tok, tag in tagged:
        if not tok.strip():
            continue
        if tag.startswith("N") or tag in {"NOUN", "PROPN"}:
            surfaces.append(tok)
        if len(surfaces) >= 12:
            break
    return surfaces


def extract_aspect_mentions(
    *,
    review_id: str,
    text: str,
    cfg: AspectExtractionConfig = AspectExtractionConfig(),
    canonical_aspects: Sequence[str] = ASPECT_CANONICAL,
) -> List[AspectMention]:
    """
    Hybrid aspect extraction:
    - sentence split
    - extract candidate surfaces (Thai/EN heuristics)
    - map surfaces to canonical aspects (keyword/fuzzy/embeddings)
    Returns aspect mentions ready for aspect-sentence sentiment scoring.
    """

    text = normalize_text(text)
    if not text:
        return []

    kw_map = aspect_keyword_map()
    mentions: List[AspectMention] = []
    sentences = split_sentences(text, cfg.sentence_split)

    for i, sent in enumerate(sentences):
        sent_norm = normalize_text(sent)
        if not sent_norm:
            continue

        if contains_thai(sent_norm):
            surfaces = _extract_candidate_surfaces_th(sent_norm, kw_map)
        else:
            surfaces = _extract_candidate_surfaces_en(sent_norm, kw_map)

        count = 0
        for surface in surfaces:
            if not surface or len(surface) < cfg.min_surface_len:
                continue
            asp = map_surface_to_aspect(surface, cfg=cfg.mapping, canonical_aspects=canonical_aspects)
            if asp is None:
                continue
            mentions.append(
                AspectMention(
                    review_id=str(review_id),
                    sentence_idx=i,
                    sentence_text=sent_norm,
                    surface=surface,
                    aspect=asp,
                )
            )
            count += 1
            if count >= cfg.max_surfaces_per_sentence:
                break

    # Dedupe mentions by (sentence_idx, aspect)
    seen = set()
    deduped: List[AspectMention] = []
    for m in mentions:
        key = (m.sentence_idx, m.aspect)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    return deduped

