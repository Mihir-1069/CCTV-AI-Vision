from pathlib import Path

from data_pipeline import DATA_PIPELINE_ROOT, PROCESSED_DATA_ROOT, RAW_DATA_ROOT, REPORTS_ROOT
from data_pipeline.checks import check_dataset_presence, dataset_status, require_dataset_dirs
from data_pipeline.eda import count_files, summarize_dataset_tree


def test_data_pipeline_package_layout_exists():
    assert DATA_PIPELINE_ROOT.exists()
    assert RAW_DATA_ROOT.exists()
    assert PROCESSED_DATA_ROOT.exists()
    assert REPORTS_ROOT.exists()


def test_dataset_directories_can_be_created():
    dirs = require_dataset_dirs()
    assert "raw" in dirs
    assert "processed" in dirs
    assert "reports" in dirs
    for value in dirs.values():
        assert Path(value).exists()


def test_dataset_status_contains_expected_paths():
    status = dataset_status()
    assert "raw" in status
    assert "processed" in status
    assert "reports" in status


def test_missing_dataset_detection_reports_missing_entries():
    result = check_dataset_presence("missing_dataset", ["images", "labels"])
    assert any("Missing dataset directory" in item for item in result)


def test_eda_helpers_are_safe_on_missing_paths():
    assert count_files("nonexistent_root") == 0
    summary = summarize_dataset_tree("nonexistent_root")
    assert summary["children"] == []
