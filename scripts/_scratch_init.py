"""
Apply scratch/temp dir before heavy deps. Import immediately after _bootstrap.

Reads --scratch_dir from sys.argv or env RRIS_SCRATCH_DIR; auto-redirects when C: is low.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

from rris.runtime_env import apply_scratch_config, parse_scratch_dir_from_argv

apply_scratch_config(_REPO_ROOT, scratch_dir=parse_scratch_dir_from_argv())
