"""CrowdHuman reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def summarize_odgt_file(annotation_file: str | Path) -> Dict[str, Any]:
    """Return a minimal summary of an ODGT annotation file without parsing full data."""
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    count = 0
    total_boxes = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            import json
            record = json.loads(line)
            count += 1
            total_boxes += len(record.get("gtboxes", []))

    return {
        "annotation_file": str(path),
        "records": count,
        "boxes": total_boxes,
    }
