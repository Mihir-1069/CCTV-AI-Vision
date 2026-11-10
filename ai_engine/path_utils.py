"""
Path Utilities for SchoolVision AI
Provides centralized path resolution for configs, models, datasets, and outputs.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory (CCTV-AI-Vision/).

    Returns:
        Path to project root
    """
    # This file is at: ai_engine/path_utils.py
    # Project root is 2 levels up
    return Path(__file__).parent.parent.resolve()


def get_config_path(config_name: str) -> Path:
    """
    Get path to a configuration file.

    Args:
        config_name: Name of config file (e.g., "model.yaml")

    Returns:
        Path to config file in configs/
    """
    return get_project_root() / "configs" / config_name


def get_models_dir() -> Path:
    """
    Get the models directory.

    Returns:
        Path to models/
    """
    return get_project_root() / "models"


def get_pretrained_models_dir() -> Path:
    """
    Get the pretrained models directory.

    Returns:
        Path to models/pretrained/
    """
    return get_models_dir() / "pretrained"


def get_trained_models_dir() -> Path:
    """
    Get the trained models directory.

    Returns:
        Path to models/trained/
    """
    return get_models_dir() / "trained"


def get_data_dir() -> Path:
    """
    Get the data directory.

    Returns:
        Path to data/
    """
    return get_project_root() / "data"


def get_videos_dir() -> Path:
    """
    Get the videos directory.

    Returns:
        Path to data/videos/
    """
    return get_data_dir() / "videos"


def get_datasets_dir() -> Path:
    """
    Get the datasets directory.

    Returns:
        Path to data/datasets/
    """
    return get_data_dir() / "datasets"


def get_output_dir() -> Path:
    """
    Get the output directory.

    Returns:
        Path to output/
    """
    return get_project_root() / "output"


def get_storage_dir() -> Path:
    """
    Get the storage directory.

    Returns:
        Path to storage/
    """
    return get_project_root() / "storage"


def get_logs_dir() -> Path:
    """
    Get the logs directory.

    Returns:
        Path to logs/
    """
    return get_project_root() / "logs"


def get_evaluation_dir() -> Path:
    """
    Get the evaluation output directory.

    Returns:
        Path to output/evaluation/
    """
    return get_output_dir() / "evaluation"


def get_evidence_dir() -> Path:
    """
    Get the evidence storage directory.

    Returns:
        Path to storage/evidence/
    """
    return get_storage_dir() / "evidence"


def get_reports_dir() -> Path:
    """
    Get the reports storage directory.

    Returns:
        Path to storage/reports/
    """
    return get_storage_dir() / "reports"


def get_annotated_videos_dir() -> Path:
    """
    Get the annotated videos storage directory.

    Returns:
        Path to storage/annotated_videos/
    """
    return get_storage_dir() / "annotated_videos"


def get_trained_model_path(model_filename: str) -> Path:
    """
    Get path to a trained model file.

    Args:
        model_filename: Filename inside models/trained/ (e.g., "anomaly_detector.joblib")

    Returns:
        Path to models/trained/<model_filename>
    """
    return get_trained_models_dir() / model_filename


def get_pretrained_model_path(model_filename: str) -> Path:
    """
    Get path to a pretrained model file.

    Args:
        model_filename: Filename inside models/pretrained/ (e.g., "yolo11n.pt")

    Returns:
        Path to models/pretrained/<model_filename>
    """
    return get_pretrained_models_dir() / model_filename
