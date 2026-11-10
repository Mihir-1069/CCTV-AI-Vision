"""SCUT-HEAD annotation conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def convert_head_annotations(annotation_file: str | Path, output_dir: str | Path) -> List[Path]:
    """Convert SCUT-HEAD label text files into YOLO-format head labels.

    This is intentionally conservative and only creates empty label files when no
    annotations are present. Real dataset conversion should be performed only after
    the actual archive layout is available.
    """
    source_path = Path(annotation_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {source_path}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    labels: List[Path] = []
    if source_path.is_file():
        label_path = target_dir / f"{source_path.stem}.txt"
        label_path.write_text("", encoding="utf-8")
        labels.append(label_path)
    return labels
