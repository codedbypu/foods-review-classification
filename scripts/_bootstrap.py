"""Make `rris` importable when running scripts without `pip install -e .`."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_LOG = _ROOT / "debug-9aac7e.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, *, run_id: str = "pre-fix") -> None:
    payload = {
        "sessionId": "9aac7e",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# #region agent log
_agent_log(
    "A",
    "_bootstrap.py:pre",
    "environment before path fix",
    {
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "src_exists": _SRC.is_dir(),
        "src_on_path_before": str(_SRC) in sys.path,
        "rris_spec_before": str(importlib.util.find_spec("rris")),
        "pip_rris_installed": importlib.util.find_spec("rris") is not None,
    },
)
# #endregion

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# #region agent log
_agent_log(
    "C",
    "_bootstrap.py:post",
    "environment after path fix",
    {
        "src_on_path_after": str(_SRC) in sys.path,
        "rris_spec_after": str(importlib.util.find_spec("rris")),
    },
)
# #endregion
