"""
Make `rris` importable when running scripts without `pip install -e .`.

COMMON ERRORS:
  - ModuleNotFoundError: No module named 'rris'
    Cause: Python cannot see src/rris unless you pip install -e . or import this module first.
    Fix: Keep `import _bootstrap` as the first import in every script under scripts/, or pip install -e .
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
