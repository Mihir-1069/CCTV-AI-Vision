"""CrowdHuman annotation conversion helpers.

CrowdHuman uses ODGT annotation files with person boxes and optional ignore flags.
The conversion here targets a YOLO-style detection layout that can later be used
for YOLO11 training preparation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class CrowdHumanConverter:
    """Convert CrowdHuman ODGT annotations to YOLO text labels."""

    def __init__(self, source_dir: str | Path, output_dir: str | Path | None = None):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir) if output_dir is not None else self.source_dir / "yolo_labels"

    def _iter_odgt_records(self, file_path: Path) -> Iterable[Dict[str, Any]]:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield record

    def convert(self, annotation_file: str | Path) -> List[Path]:
        """Convert an ODGT file to YOLO label files and return generated paths."""
        annotation_path = Path(annotation_file)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written_files: List[Path] = []

        for record in self._iter_odgt_records(annotation_path):
            image_name = record.get("ID") or record.get("image_id")
            if not image_name:
                continue

            boxes = record.get("gtboxes", [])
            labels: List[str] = []
            for box in boxes:
                if box.get("ignore", 0) == 1:
                    continue
                if box.get("fbox") is None and box.get("head_attr") is None:
                    continue
                x, y, w, h = box.get("fbox", [0, 0, 0, 0])
                if w <= 0 or h <= 0:
                    continue
                labels.append(f"0 {x + w / 2} {y + h / 2} {w} {h}")

            label_path = self.output_dir / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(labels), encoding="utf-8")
            written_files.append(label_path)

        return written_files
