# SchoolVision AI — CCTV Intelligence Platform

An AI-powered CCTV monitoring platform for school environments. Detects and
tracks people, estimates poses, recognises activities and raises configurable
alerts — all running on CPU without a dedicated GPU.

> **Privacy note:** All tracking uses anonymous temporary IDs. No facial
> recognition is performed. No biometric data is stored. Tracking IDs reset
> on every pipeline restart.

---

## Repository structure

```
CCTV-AI-Vision/
├── configs/                   # Runtime configuration (YAML)
│   ├── model.yaml             # Detection, pose, tracking, activity thresholds
│   ├── camera.yaml            # Per-camera zones, counting lines, occupancy limits
│   ├── app.yaml               # Logging, output paths, integration flags
│   └── alert_rules.yaml       # Rule-engine conditions and severities
│
├── data/
│   ├── videos/                # Input video files for testing
│   ├── rtsp/                  # RTSP source definitions (not committed)
│   └── datasets/              # Raw and processed annotation datasets
│       ├── detection/
│       ├── tracking/
│       ├── activity/
│       └── anomaly/
│
├── models/
│   ├── pretrained/            # Downloaded pretrained weights (not committed)
│   │   ├── yolo11n.pt         # YOLO11n — person detection
│   │   └── yolo11n-pose.pt    # YOLO11n-pose — keypoint estimation
│   └── trained/               # Project-trained weights (not committed)
│
├── data_pipeline/             # Dataset preparation utilities (offline use only)
│   ├── crowdhuman/            # CrowdHuman annotation converter
│   ├── mot17/                 # MOT17 annotation converter
│   ├── scut_head/             # SCUT-HEAD annotation converter
│   └── common/                # Shared conversion helpers
│
├── video_processing/          # Video I/O and frame utilities
│   ├── video_reader.py        # VideoReader — context-managed frame extraction
│   ├── frame_processor.py     # FrameProcessor — resize, skip, validate
│   └── video_utils.py         # VideoUtils — synthetic video, aspect ratio
│
├── ai_engine/                 # Core CV and AI inference modules
│   ├── path_utils.py          # Centralised project-root path resolution
│   ├── detection/             # PersonDetector (YOLO11n)
│   ├── tracking/              # ByteTrackManager + SimpleIOUTracker fallback
│   ├── pose/                  # PoseEstimator, PoseUtils, PoseAnalyzer
│   ├── activity/              # ActivityRecognizer (rule-based, geometric)
│   ├── anomaly/               # AnomalyDetector (autoencoder, scikit-learn)
│   ├── crowd/                 # CrowdAnalytics
│   ├── analytics/             # OccupancyManager, EntryExitCounter, ZoneManager
│   ├── inference/             # InferencePipeline — reusable per-frame wrapper
│   └── training/              # Training stubs (not yet implemented)
│
├── event_engine/              # Rule evaluation and alert generation
│   ├── rule_engine.py         # RuleEngine — evaluates alert_rules.yaml
│   ├── alert_generator.py     # AlertGenerator — formats and routes alerts
│   └── event_correlator.py    # EventCorrelator — deduplicates alert streams
│
├── backend/                   # FastAPI backend (future integration)
├── database/                  # Database layer (future integration)
├── frontend/                  # React dashboard (future integration)
│
├── storage/                   # Operational outputs (not committed)
│   ├── evidence/              # Saved frames linked to alerts
│   ├── reports/               # Generated PDF/JSON reports
│   └── annotated_videos/      # Processed video outputs
│
├── output/
│   └── evaluation/            # Evaluation metrics and plots
│
├── logs/                      # Runtime log files (not committed)
├── docs/                      # Architecture and API documentation
├── scripts/                   # CLI entry points
│   └── run_video.py           # Main pipeline runner
├── tests/                     # Unit and integration tests (301 passing)
├── requirements.txt
└── pytest.ini
```

---

## Pipeline

```
VideoReader
    ↓
FrameProcessor
    ↓
PersonDetector           ← YOLO11n (pretrained, CPU)
    ↓
ByteTrackManager         ← SimpleIOUTracker fallback when BYTETracker unavailable
    ├── OccupancyManager
    ├── EntryExitCounter
    ├── ZoneManager
    ├── PoseEstimator    ← YOLO11n-pose (pretrained, CPU)
    │       ↓
    │   ActivityRecognizer  ← Rule-based geometric classifier
    │       ↓
    │   AnomalyDetector     ← Autoencoder reconstruction error (untrained by default)
    └── CrowdAnalytics
              ↓
         RuleEngine
              ↓
         AlertGenerator
              ↓
         EventCorrelator
              ↓
         Outputs (annotated video, JSON, storage/)
```

### Running the pipeline

```powershell
# From the project root
python scripts/run_video.py --source data/videos/test_sample.mp4

# With optional flags
python scripts/run_video.py --source data/videos/sample.mp4 `
    --camera-id cam_001 --save output/annotated_videos/out.mp4 --show
```

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Models are downloaded automatically on first run from Ultralytics Hub,
# or place them manually in models/pretrained/:
#   models/pretrained/yolo11n.pt
#   models/pretrained/yolo11n-pose.pt
```

---

## Running tests

```powershell
python -m pytest tests/ -q
# Expected: 301 passed
```

---

## Module responsibilities

| Module | Class | Status |
|--------|-------|--------|
| `video_processing` | `VideoReader`, `FrameProcessor`, `VideoUtils` | Implemented |
| `ai_engine/detection` | `PersonDetector` | Implemented — YOLO11n |
| `ai_engine/tracking` | `ByteTrackManager` | Implemented — SimpleIOUTracker fallback |
| `ai_engine/pose` | `PoseEstimator`, `PoseUtils`, `PoseAnalyzer` | Implemented — YOLO11n-pose |
| `ai_engine/activity` | `ActivityRecognizer` | Implemented — rule-based |
| `ai_engine/anomaly` | `AnomalyDetector` | Implemented — **model not trained** |
| `ai_engine/crowd` | `CrowdAnalytics` | Implemented |
| `ai_engine/analytics` | `OccupancyManager`, `EntryExitCounter`, `ZoneManager` | Implemented |
| `ai_engine/inference` | `InferencePipeline` | Implemented — reusable frame wrapper |
| `event_engine` | `RuleEngine`, `AlertGenerator`, `EventCorrelator` | Implemented |
| `data_pipeline` | Dataset converters | Implemented — no datasets included |
| `backend` | FastAPI routes | Scaffold only |
| `database` | Models and migrations | Scaffold only |
| `frontend` | React dashboard | Scaffold only |

### Model status

| Model | File | Status |
|-------|------|--------|
| YOLO11n (detection) | `models/pretrained/yolo11n.pt` | Pretrained — COCO |
| YOLO11n-pose | `models/pretrained/yolo11n-pose.pt` | Pretrained — COCO |
| Anomaly autoencoder | `models/trained/anomaly/anomaly_detector.joblib` | **Not trained** — returns score 0.0 until trained |

The anomaly detector runs in graceful fallback mode when no trained weights
are found. To train it, collect normal-condition pose data and run
`scripts/train_anomaly.py` (not yet implemented).

---

## Dataset storage conventions

| Content | Location |
|---------|----------|
| Raw downloaded datasets | `data/datasets/{detection,tracking,activity,anomaly}/` |
| Dataset preparation scripts | `data_pipeline/` |
| Converted/processed annotations | `data_pipeline/processed/` |
| Dataset validation reports | `data_pipeline/reports/` |

Datasets are **not committed** to this repository. Download them separately
and place them in the appropriate `data/datasets/` subdirectory.

---

## Path management

All project-root-relative paths are resolved via `ai_engine/path_utils.py`:

```python
from ai_engine.path_utils import (
    get_project_root,
    get_config_path,          # configs/<name>
    get_pretrained_models_dir, # models/pretrained/
    get_trained_models_dir,    # models/trained/
    get_videos_dir,            # data/videos/
    get_datasets_dir,          # data/datasets/
    get_logs_dir,              # logs/
    get_evidence_dir,          # storage/evidence/
    get_reports_dir,           # storage/reports/
    get_annotated_videos_dir,  # storage/annotated_videos/
    get_evaluation_dir,        # output/evaluation/
)
```

Hardcoded absolute paths have been removed from all runtime modules.
