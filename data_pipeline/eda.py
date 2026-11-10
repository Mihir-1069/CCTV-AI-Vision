"""Dataset exploratory analysis utilities.

These functions are intentionally lightweight and support dataset readiness checks
without requiring actual downloads or model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def count_files(root: str | Path, suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> int:
    """Count files with the given suffixes under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return 0
    return sum(1 for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def summarize_dataset_tree(root: str | Path) -> Dict[str, List[str]]:
    """Return a summary of immediate children for a dataset root."""
    root_path = Path(root)
    if not root_path.exists():
        return {"root": str(root_path), "children": []}

    children = sorted(path.name for path in root_path.iterdir())
    return {"root": str(root_path), "children": children}
