"""MOT17 tracking evaluation helpers.

These helpers are intentionally lightweight and only prepare metadata for later
TrackEval integration. No training or inference is triggered here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def summarize_sequence_dir(sequence_dir: str | Path) -> Dict[str, Any]:
    """Summarize a MOT17 sequence directory structure."""
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")

    files = sorted(p.name for p in sequence_path.iterdir())
    return {
        "sequence_dir": str(sequence_path),
        "items": files,
        "image_count": sum(1 for item in files if item.lower().endswith((".jpg", ".png"))),
    }
