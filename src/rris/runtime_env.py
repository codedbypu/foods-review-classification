"""
Process scratch / temp directory (Windows low C: space, XGBoost bad_malloc).

COMMON ERRORS:
  - bad_malloc: often low C: free space (pagefile). Use --scratch_dir on a drive with space,
    or env RRIS_SCRATCH_DIR (use a drive that exists, e.g. F:\\rris-scratch — not D: if you have no D:)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

ENV_SCRATCH = "RRIS_SCRATCH_DIR"
_SCRATCH_ENV_KEYS = ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LOKY_TEMP")


def _path_free_bytes(path: Path) -> Optional[int]:
    try:
        return int(shutil.disk_usage(path.anchor if path.anchor else path).free)
    except OSError:
        return None


def _system_drive_free_bytes() -> Optional[int]:
    if os.name != "nt":
        return None
    return _path_free_bytes(Path("C:\\"))


def _windows_drive_hints() -> str:
    import string

    parts: list[str] = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:\\")
        if not root.exists():
            continue
        try:
            gb = shutil.disk_usage(root).free / 1e9
            parts.append(f"{letter}: ({gb:.1f} GB free)")
        except OSError:
            parts.append(f"{letter}:")
    return ", ".join(parts) if parts else "(none detected)"


def _ensure_scratch_path_exists(p: Path) -> None:
    """Fail fast with a clear message if the drive/path is missing (e.g. D: on a PC with only C/E/F)."""
    if os.name == "nt" and p.drive:
        drive_root = Path(p.drive + "\\")
        if not drive_root.exists():
            raise FileNotFoundError(
                f"Scratch path drive does not exist: {p.drive!r} "
                f"(full path {p!s}). Available drives on this PC: {_windows_drive_hints()}. "
                f'Use an existing drive, e.g. --scratch_dir "F:\\rris-scratch".'
            )
    try:
        p.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot create scratch directory {p!s}: {e}. "
            f"Available drives: {_windows_drive_hints()}."
        ) from e


def configure_scratch_dir(
    scratch_dir: Union[str, Path],
    *,
    force: bool = True,
    set_hf_home: bool = True,
) -> Path:
    """
    Point process temp folders to scratch_dir (user-chosen drive/path).
    Call before heavy imports (xgboost/joblib/torch) when possible.
    """
    p = Path(scratch_dir).expanduser()
    # resolve() only after drive exists — missing D: would raise on resolve()
    _ensure_scratch_path_exists(p)
    p = p.resolve()
    free = _path_free_bytes(p)
    for key in _SCRATCH_ENV_KEYS:
        if force or not os.environ.get(key):
            os.environ[key] = str(p)
    if set_hf_home and (force or not os.environ.get("HF_HOME")):
        hf = p / "huggingface"
        hf.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf)
    logger.info(
        "Scratch dir %s (%.2f GB free on volume); TEMP=%s",
        p,
        (free / 1e9) if free is not None else -1.0,
        os.environ.get("TEMP"),
    )
    return p


def scratch_dir_from_env() -> Optional[Path]:
    raw = os.environ.get(ENV_SCRATCH, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def apply_scratch_config(
    repo_root: Path,
    *,
    scratch_dir: Optional[Union[str, Path]] = None,
    min_free_gb: float = 2.0,
) -> Path:
    """
    Priority: explicit scratch_dir > RRIS_SCRATCH_DIR env > auto if C: low else repo/.cache/tmp.
    Returns the scratch path in use (may be configured or only the default path).
    """
    if scratch_dir is not None and str(scratch_dir).strip():
        return configure_scratch_dir(scratch_dir, force=True)

    from_env = scratch_dir_from_env()
    if from_env is not None:
        return configure_scratch_dir(from_env, force=True)

    free_c = _system_drive_free_bytes()
    if free_c is not None and free_c >= min_free_gb * (1024**3):
        return repo_root / ".cache" / "tmp"

    scratch = repo_root / ".cache" / "tmp"
    configure_scratch_dir(scratch, force=True)
    if free_c is not None:
        logger.warning(
            "Low space on C: (%.2f GB free); using scratch under repo: %s. "
            "For another drive use --scratch_dir D:\\your\\folder or set %s.",
            free_c / 1e9,
            scratch,
            ENV_SCRATCH,
        )
    return scratch


def parse_scratch_dir_from_argv(argv: Optional[list[str]] = None) -> Optional[str]:
    """Read --scratch_dir from argv without consuming other flags (for early bootstrap)."""
    import sys

    argv = list(argv if argv is not None else sys.argv)
    out: Optional[str] = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--scratch_dir" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
            continue
        if tok.startswith("--scratch_dir="):
            out = tok.split("=", 1)[1]
        i += 1
    return out


def add_scratch_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default=None,
        help=(
            "Folder on a drive with free space for TEMP/joblib/HF cache. "
            f"Permanent default: env {ENV_SCRATCH}. Applied at process start."
        ),
    )


def apply_scratch_from_args(
    repo_root: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    """Re-apply if main() parsed --scratch_dir (early argv parse may already have set it)."""
    raw = getattr(args, "scratch_dir", None)
    if raw and str(raw).strip():
        return configure_scratch_dir(raw, force=True)
    return None
