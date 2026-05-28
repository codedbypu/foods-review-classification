"""
Aspect keyword/fuzzy/embedding mapping for extracted surfaces.

COMMON ERRORS (kernel crash / disk full):
  Previously SentenceTransformer was instantiated on EVERY map_surface_to_aspect call.
  score_and_flag on 500+ reviews could try to download ~470MB hundreds of times.
  Fix: @lru_cache on _get_embedding_model + use --skip_aspects in score_and_flag for notebooks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _get_embedding_model(model_name: str):
    """Load embedding model once per process (critical for score_and_flag loops)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


ASPECT_CANONICAL: Tuple[str, ...] = (
    "food",
    "service",
    "price",
    "ambience",
    "cleanliness",
    "location",
    "delivery",
)


_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")


def _normalize_surface(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# High-precision keyword mapping first (EN/TH). Keep these conservative.
_KEYWORD_TO_ASPECT: Dict[str, str] = {
    # food
    "food": "food",
    "taste": "food",
    "flavor": "food",
    "menu": "food",
    "portion": "food",
    "dish": "food",
    "delicious": "food",
    "not delicious": "food",
    "รสชาติ": "food",
    "อาหาร": "food",
    "เมนู": "food",
    "วัตถุดิบ": "food",
    "ปริมาณ": "food",
    "อร่อย": "food",
    "ไม่อร่อย": "food",
    # service
    "service": "service",
    "staff": "service",
    "waiter": "service",
    "waitress": "service",
    "server": "service",
    "attitude": "service",
    "rude": "service",
    "friendly": "service",
    "พนักงาน": "service",
    "บริการ": "service",
    "มารยาท": "service",
    "พูดจา": "service",
    "รอ": "service",
    # price
    "price": "price",
    "cost": "price",
    "value": "price",
    "expensive": "price",
    "cheap": "price",
    "ราคา": "price",
    "แพง": "price",
    "ถูก": "price",
    "คุ้ม": "price",
    # ambience
    "ambience": "ambience",
    "atmosphere": "ambience",
    "music": "ambience",
    "noise": "ambience",
    "decor": "ambience",
    "vibe": "ambience",
    "บรรยากาศ": "ambience",
    "เพลง": "ambience",
    "เสียงดัง": "ambience",
    "ตกแต่ง": "ambience",
    # cleanliness
    "clean": "cleanliness",
    "dirty": "cleanliness",
    "hygiene": "cleanliness",
    "restroom": "cleanliness",
    "toilet": "cleanliness",
    "สะอาด": "cleanliness",
    "สกปรก": "cleanliness",
    "สุขอนามัย": "cleanliness",
    "ห้องน้ำ": "cleanliness",
    # location
    "location": "location",
    "parking": "location",
    "near": "location",
    "far": "location",
    "access": "location",
    "ที่จอดรถ": "location",
    "เดินทาง": "location",
    "ใกล้": "location",
    "ไกล": "location",
    "ทำเล": "location",
    # delivery
    "delivery": "delivery",
    "grab": "delivery",
    "lineman": "delivery",
    "linemanfood": "delivery",
    "packaging": "delivery",
    "delivery time": "delivery",
    "ส่ง": "delivery",
    "เดลิเวอรี่": "delivery",
    "ไรเดอร์": "delivery",
    "แพ็กเกจ": "delivery",
    "แพคเกจ": "delivery",
    "กล่อง": "delivery",
}


@dataclass(frozen=True)
class AspectMappingConfig:
    """
    Aspect mapping strategy:
    - First try exact/keyword matches (high precision).
    - Then fuzzy match (still local, fast).
    - Optionally fallback to embeddings for multilingual semantic matching.
    """

    fuzzy_threshold: int = 88  # rapidfuzz similarity 0..100
    enable_embeddings_fallback: bool = True
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def map_surface_to_aspect(
    surface: str,
    *,
    cfg: AspectMappingConfig = AspectMappingConfig(),
    canonical_aspects: Sequence[str] = ASPECT_CANONICAL,
) -> Optional[str]:
    """
    Map an extracted surface phrase to a canonical aspect string.
    Returns None if no confident mapping is found.
    """

    s = _normalize_surface(surface)
    if not s:
        return None

    # 1) exact keyword match
    if s in _KEYWORD_TO_ASPECT:
        return _KEYWORD_TO_ASPECT[s]

    # 2) substring keyword match (Thai can be concatenated)
    for kw, asp in _KEYWORD_TO_ASPECT.items():
        if kw and kw in s:
            return asp

    # 3) fuzzy match against keywords (helps with typos/romanization)
    try:
        from rapidfuzz import fuzz
    except Exception:
        fuzz = None

    if fuzz is not None:
        best_kw = None
        best_score = -1
        for kw in _KEYWORD_TO_ASPECT.keys():
            score = fuzz.token_set_ratio(s, kw)
            if score > best_score:
                best_kw = kw
                best_score = score
        if best_kw is not None and best_score >= cfg.fuzzy_threshold:
            return _KEYWORD_TO_ASPECT[best_kw]

    # 4) embeddings fallback — needs HF model + disk; disable in notebooks via AspectMappingConfig
    if cfg.enable_embeddings_fallback:
        try:
            import numpy as np
        except Exception as e:
            logger.warning("Embeddings fallback unavailable (%s). Returning None.", e)
            return None

        try:
            model = _get_embedding_model(cfg.embedding_model_name)
            candidates = list(canonical_aspects)
            emb = model.encode([s] + candidates, normalize_embeddings=True)
            q = emb[0]
            cand = emb[1:]
            sims = (cand @ q).astype(float)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            # conservative threshold; tune on validation
            if best_sim >= 0.45:
                return candidates[best_idx]
        except Exception as e:
            logger.warning("Embeddings mapping failed (%s). Returning None.", e)
            return None

    return None


def aspect_keyword_map() -> Dict[str, str]:
    """Public accessor for the keyword→aspect mapping (do not mutate)."""
    return dict(_KEYWORD_TO_ASPECT)

