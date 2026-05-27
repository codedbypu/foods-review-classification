from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


def default_paths(repo_root: Path) -> Paths:
    data_dir = repo_root / "data"
    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
        models_dir=repo_root / "models",
        reports_dir=repo_root / "reports",
    )

