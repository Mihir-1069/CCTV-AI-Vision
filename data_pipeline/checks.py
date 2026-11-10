"""Dataset validation utilities.

These helpers validate the existence of expected datasets without downloading
anything or modifying the running inference pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import DATA_PIPELINE_ROOT, PROCESSED_DATA_ROOT, RAW_DATA_ROOT, REPORTS_ROOT


def dataset_root() -> Path:
    """Return the package root used for dataset preparation."""
    return DATA_PIPELINE_ROOT


def ensure_directory(path: Path) -> Path:
    """Create and return a directory path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_dataset_dirs() -> Dict[str, Path]:
    """Ensure all managed dataset directories exist."""
    dirs = {
        "raw": ensure_directory(RAW_DATA_ROOT),
        "processed": ensure_directory(PROCESSED_DATA_ROOT),
        "reports": ensure_directory(REPORTS_ROOT),
    }
    return dirs


def check_dataset_presence(dataset_name: str, expected_files: Optional[Iterable[str]] = None) -> List[str]:
    """Return a list of missing dataset entries for a dataset name.

    Args:
        dataset_name: dataset folder name stored under raw/
        expected_files: optional path fragments expected under the dataset.
    """
    dataset_dir = RAW_DATA_ROOT / dataset_name
    errors: List[str] = []

    if not dataset_dir.exists():
        errors.append(f"Missing dataset directory: {dataset_dir}")
        return errors

    if expected_files is None:
        return errors

    for relative_path in expected_files:
        candidate = dataset_dir / relative_path
        if not candidate.exists():
            errors.append(f"Missing expected file or folder: {candidate}")

    return errors


def dataset_status() -> Dict[str, object]:
    """Provide a simple dataset status summary for the managed folders."""
    dirs = require_dataset_dirs()
    return {
        "data_pipeline_root": str(DATA_PIPELINE_ROOT),
        "raw": str(dirs["raw"]),
        "processed": str(dirs["processed"]),
        "reports": str(dirs["reports"]),
    }
