"""Dataset preparation and evaluation pipeline for SchoolVision AI.

This package is intentionally separated from the runtime CV inference stack so that
annotation conversion, dataset validation and training-prep utilities can evolve
without touching the existing detection/tracking activity pipeline.
"""

from pathlib import Path

DATA_PIPELINE_ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = DATA_PIPELINE_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_PIPELINE_ROOT / "processed"
REPORTS_ROOT = DATA_PIPELINE_ROOT / "reports"

__all__ = [
    "DATA_PIPELINE_ROOT",
    "RAW_DATA_ROOT",
    "PROCESSED_DATA_ROOT",
    "REPORTS_ROOT",
]
