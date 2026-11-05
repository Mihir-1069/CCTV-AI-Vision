# Computer Vision Pipeline Setup Guide

**Person 1 - Computer Vision and AI Pipeline Development**

This document provides setup instructions for the Computer Vision Pipeline components of SchoolVision AI.

---

## Prerequisites

- Python 3.12 or higher
- Windows, Linux, or macOS
- No GPU required (CPU inference supported)
- 8GB RAM minimum (16GB recommended)

---

## Installation Steps

### 1. Create Virtual Environment (Windows PowerShell)

```powershell
# Navigate to project root
cd d:\Projects\cv_school

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If activation fails due to execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
# Ensure pip is upgraded
python -m pip install --upgrade pip

# Install CV pipeline dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```powershell
# Check Python version
python --version

# Check installed packages
pip list

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"

# Test NumPy
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
```

### 4. Create Directory Structure

```powershell
# Create required directories
mkdir data\videos
mkdir data\rtsp
mkdir models\pretrained
mkdir output\annotated_videos
mkdir output\json
mkdir output\evaluation
mkdir output\plots
mkdir logs
```

### 5. Download Pretrained Models (Automatic)

The pipeline will automatically download YOLO11n and YOLO11n-pose models on first run.

To download manually:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
```

Models will be saved to: `C:\Users\<username>\AppData\Roaming\Ultralytics\`

---

## Configuration Files

Configuration files are located in `CCTV-AI-Vision/configs/`:

- **model.yaml**: YOLO model settings, tracking parameters, performance options
- **camera.yaml**: Camera zones, counting lines, occupancy settings
- **app.yaml**: Application paths, logging, output settings

Edit these files to customize:
- Detection confidence thresholds
- Camera zones and counting lines
- Output directories
- Processing parameters

---

## Running the Pipeline

### Process a Video File

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Navigate to scripts directory
cd CCTV-AI-Vision\scripts

# Run video processing (after implementation)
python run_video.py --video data/videos/sample.mp4 --camera cam_001
```

### Process RTSP Stream

```powershell
# Run camera processing (after implementation)
python run_camera.py --rtsp rtsp://camera_ip/stream --camera cam_001
```

---

## Testing

```powershell
# Run unit tests (after implementation)
cd CCTV-AI-Vision
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ai_engine --cov=analytics --cov-report=html
```

---

## Module Overview

### Module A: Video Processing ✓ (COMPLETED)
- Location: `video_processing/video_reader.py`, `video_processing/frame_processor.py`, `video_processing/video_utils.py`
- Function: Read videos, extract frames, handle metadata, resize frames, validate frames
- Features:
  - VideoReader: Context manager for video I/O, metadata extraction, frame reading
  - FrameProcessor: Frame resizing, validation, preprocessing, frame skipping
  - VideoUtils: Synthetic video generation, video info, aspect ratio calculation
- Testing: `tests/test_video_reader.py`, `tests/test_frame_processor.py`
- Demo: `scripts/test_video_processing.py`

### Module B: YOLO Detection
- Location: `ai_engine/detection/yolo_detector.py`
- Function: Detect people using YOLO11n

### Module C: ByteTrack Tracking ✓ (COMPLETED)
- Location: `ai_engine/tracking/bytetrack_manager.py`, `ai_engine/tracking/tracking_utils.py`
- Function: Anonymous tracking of detected people using ByteTrack or SimpleIOUTracker
- Features:
  - ByteTrackManager: Multi-object tracking, track history, active track management
  - SimpleIOUTracker: Fallback IoU-based tracker when ByteTrack unavailable
  - TrackingUtils: Detection format conversion, track ID drawing, IoU calculation
  - Track history tracking per track ID
  - Track age calculation
  - Performance tracking (tracking latency)
  - Integration with configs/model.yaml
- Testing: `tests/test_bytetrack.py`
- Demo: `scripts/test_bytetrack.py`

### Module D: Occupancy Analytics
- Location: `analytics/occupancy.py`
- Function: Count people in camera zones

### Module E: Entry/Exit Detection
- Location: `analytics/entry_exit.py`
- Function: Count people crossing virtual lines

### Module F: Pose Estimation ✓ (COMPLETED)
- Location: `ai_engine/pose/pose_estimator.py`, `ai_engine/pose/pose_utils.py`, `ai_engine/pose/pose_analyzer.py`
- Function: Extract human body keypoints using YOLO11n-pose
- Features:
  - PoseEstimator: COCO 17 keypoint extraction, bbox + confidence, CPU inference
  - PoseUtils: Keypoint conversion, skeleton drawing, greedy IoU tracking association
  - PoseAnalyzer: Geometric joint angles (elbow, knee, hip, shoulder), body orientation
  - One-to-one pose-to-track association with matched/unmatched/uncertain status
  - JSON-serializable output with all 17 COCO keypoints
- Testing: `tests/test_pose_estimator.py` (58 mocked unit tests + 5 model smoke tests)
- Demo: `scripts/test_pose.py`

### Module G: Performance Evaluation
- Location: `ai_engine/training/evaluate_models.py`
- Function: Measure precision, recall, FPS, latency

### Module H: JSON Output
- Location: `ai_engine/output/json_output.py`
- Function: Generate standardized JSON for backend

---

## Hardware Considerations

### CPU Inference (Default)
- Uses YOLO11n (nano) model
- Resolution: 640x480 recommended
- Expected FPS: 5-15 on modern CPU
- No GPU required

### GPU Inference (Optional)
- Set `device: "cuda"` in `configs/model.yaml`
- Requires NVIDIA GPU with CUDA support
- Expected FPS: 30-60+ depending on GPU

---

## Troubleshooting

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### OpenCV Issues
```powershell
# Uninstall and reinstall OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Model Download Failures
```powershell
# Check internet connection
# Manually download models from Ultralytics Hub
# Place in models/pretrained/ directory
```

### Memory Issues
- Reduce `resize_width` and `resize_height` in `configs/model.yaml`
- Increase `frame_skip` to process fewer frames
- Close other applications

---

## Development Workflow

1. **Phase 1**: Project Setup ✓ (COMPLETED)
2. **Phase 2**: Video Processing Module ✓ (COMPLETED)
3. **Phase 3**: YOLO Detection Module ✓ (COMPLETED)
4. **Phase 4**: ByteTrack Tracking Module ✓ (COMPLETED)
5. **Phase 5**: Occupancy Analytics Module ✓ (COMPLETED)
6. **Phase 6**: Pose Estimation Module ✓ (COMPLETED)
7. **Phase 7**: Performance Evaluation Module (PENDING)
8. **Phase 8**: JSON Output Module (PENDING)
9. **Phase 9**: Integration Scripts (PENDING)
10. **Phase 10**: Integration Tests & Documentation (PENDING)

---

## Notes

- This pipeline does NOT perform facial recognition
- All tracking is anonymous and temporary
- No personal data is stored or transmitted
- Models are pretrained; no training required for basic use
- CPU inference is fully supported
- Future GPU deployment is supported via configuration

---

## Next Steps

After completing Phase 5, proceed to Phase 6: Pose Estimation Module.

---

## Phase 5: Occupancy Analytics Module - Implementation Details

### Files Created
- `ai_engine/analytics/__init__.py` - Analytics module initialization
- `ai_engine/analytics/analytics_utils.py` - Analytics utilities (252 lines)
- `ai_engine/analytics/occupancy_manager.py` - OccupancyManager class (197 lines)
- `ai_engine/analytics/entry_exit_counter.py` - EntryExitCounter class (195 lines)
- `ai_engine/analytics/zone_manager.py` - ZoneManager class (183 lines)
- `tests/test_analytics.py` - Unit tests for analytics (432 lines)
- `scripts/test_analytics.py` - Demo script for analytics (318 lines)

### Files Modified
- `ai_engine/__init__.py` - Added analytics module exports

### OccupancyManager Class
- Tracks current number of people in camera view
- Unique active track count (avoids duplicate counting)
- Occupancy history over time (last 1000 entries)
- Maximum occupancy observed
- Minimum occupancy observed
- Average occupancy calculation
- Timestamp and frame number for every update
- Handles people temporarily disappearing from detection
- Configurable capacity and alert thresholds
- Supports camera ID for multi-camera scenarios

### EntryExitCounter Class
- Virtual line crossing detection using configurable line coordinates
- Detects when tracked person's bounding-box center crosses the line
- Determines crossing direction (entry/exit/none)
- Counts entry and exit separately
- Bidirectional, entry-only, or exit-only modes
- Configurable debounce/cooldown logic (default 10 frames)
- Avoids duplicate counts when person remains near line
- Handles person crossing back in opposite direction
- Maintains separate counts per camera instance
- Does not count newly appearing tracks as entries without actual crossing

### ZoneManager Class
- Polygon-based occupancy zones
- Configurable polygon coordinates
- Determines if person's bounding-box center is inside zone
- Counts people in each zone
- Supports multiple zones per camera
- Updates zone occupancy as tracks move
- Avoids duplicate counting within zone
- Supports zone names and IDs
- Zone track position tracking

### AnalyticsUtils Class
- Bounding-box center calculation
- Point-to-line side calculation (cross product)
- Line-crossing detection (entry/exit/none)
- Point-in-polygon detection (ray casting algorithm)
- Timestamp formatting (ISO 8601)
- Analytics output validation
- Euclidean distance calculation
- Point-to-line distance calculation

### Standardized Analytics Output Format
```python
{
  "camera_id": "camera_01",
  "timestamp": "2026-11-05T18:30:00",
  "frame_number": 150,
  "occupancy": {
    "current_count": 5,
    "max_count": 8,
    "min_count": 0,
    "average_count": 4.5
  },
  "entry_exit": {
    "total_entries": 12,
    "total_exits": 7,
    "net_count": 5
  },
  "zones": [
    {
      "zone_id": "zone_001",
      "zone_name": "main_area",
      "count": 3,
      "track_ids": [1, 2, 3]
    }
  ]
}
```

### Running Tests
```powershell
# All analytics tests
python -m pytest tests/test_analytics.py -v

# Specific test classes
python -m pytest tests/test_analytics.py::TestAnalyticsUtils -v
python -m pytest tests/test_analytics.py::TestOccupancyManager -v
python -m pytest tests/test_analytics.py::TestEntryExitCounter -v
python -m pytest tests/test_analytics.py::TestZoneManager -v
```

### Running Demo Script
```powershell
# Basic usage
python scripts/test_analytics.py --source data/videos/test_sample.mp4

# With custom parameters
python scripts/test_analytics.py --source data/videos/test_sample.mp4 --camera-id cam_001 --confidence 0.5 --imgsz 640 --device cpu --output output/annotated_videos

# With display
python scripts/test_analytics.py --source data/videos/test_sample.mp4 --display

# Limit frames
python scripts/test_analytics.py --source data/videos/test_sample.mp4 --max-frames 100
```

### Configuration Integration
The analytics modules integrate with:
- `configs/camera.yaml` - Camera zones, counting lines, occupancy settings
- `configs/app.yaml` - Output paths, timestamp format

### Test Results
- AnalyticsUtils tests: 14/14 passed
- OccupancyManager tests: 8/8 passed
- EntryExitCounter tests: 6/6 passed
- ZoneManager tests: 7/7 passed
- Total: 35/35 tests passed

### Performance Metrics (Synthetic Video Test)
**Note**: Synthetic video contains no actual people, so 0 occupancy, 0 entries/exits is correct behavior.

- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Processed frames: 30
- Total detections: 0 (expected - synthetic video has no people)
- Current occupancy: 0
- Max occupancy: 0
- Average occupancy: 0.00
- Total entries: 0
- Total exits: 0
- Net count: 0
- Zone occupancy: 0 people in main_area
- Average detection latency: 538.73ms
- Average tracking latency: 0.02ms
- Processing FPS: 1.82

### Real Video Validation
**Important**: To validate analytics on actual human-containing video:

1. Download a legally usable test video with people (e.g., from Pexels or Pixabay)
2. Place in `data/videos/` directory
3. Run analytics:
```powershell
python scripts/test_analytics.py --source data/videos/your_video.mp4 --output output/annotated_videos
```
4. Check:
   - People are detected, tracked, and counted
   - Occupancy changes as people appear/disappear
   - Entry/exit counts increment when crossing the virtual line
   - Zone counts update as people move in/out of zones
   - Annotated video shows zones and counting line
   - Analytics report and history are saved

### Limitations
- CPU inference is slow (~1.8 FPS with detection + tracking + analytics)
- Synthetic video cannot validate analytics behavior
- Requires actual human-containing video for real validation
- Entry/exit line is hardcoded in demo (should be configurable from camera.yaml)
- Zone polygon is hardcoded in demo (should be configurable from camera.yaml)
- No GPU available on current system
- Track IDs are temporary and reset on tracker reset

### Analytics Information
- **Occupancy**: Current count, max, min, average over time
- **Entry/Exit**: Virtual line crossing detection with debounce
- **Zones**: Polygon-based zones with per-zone counting
- **Track IDs**: Temporary, reset on tracker reset
- **Debounce**: 10 frames (configurable)
- **History**: Last 1000 entries for occupancy

---

## Phase 4: ByteTrack Multi-Object Tracking Module - Implementation Details

### Files Created
- `ai_engine/tracking/__init__.py` - Tracking module initialization
- `ai_engine/tracking/bytetrack_manager.py` - ByteTrackManager class (502 lines)
- `ai_engine/tracking/tracking_utils.py` - Tracking utilities (261 lines)
- `tests/test_bytetrack.py` - Unit tests for ByteTrack (345 lines)
- `scripts/test_bytetrack.py` - Demo script for tracking (248 lines)

### Files Modified
- `ai_engine/__init__.py` - Added ByteTrackManager export

### ByteTrackManager Class
- Integrates with Supervision BYTETracker when available
- Falls back to SimpleIOUTracker when BYTETracker unavailable (current situation)
- Receives detections from PersonDetector
- Associates detections across consecutive frames
- Assigns temporary tracking IDs
- Maintains track history per track ID
- Handles temporarily missed detections (track buffer)
- Handles newly appearing people
- Handles disappearing tracks
- Supports tracker reset
- Performance tracking (tracking latency)
- Track age calculation
- Integration with configs/model.yaml

### SimpleIOUTracker Class (Fallback)
- Simple IoU-based tracker when ByteTrack unavailable
- Matches detections using IoU threshold
- Maintains track age
- Removes tracks after max_age frames without detection
- Provides same interface as ByteTrack for compatibility

### TrackingUtils Class
- Convert detections to supervision format
- Convert supervision results to tracking output
- Draw track IDs on frames
- Draw tracking info overlay
- Calculate track age
- Filter tracks by age
- Calculate IoU for bounding boxes

### Tracking Output Format
```python
[
    {
        "track_id": 1,
        "class_id": 0,
        "class_name": "person",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.94,
        "track_state": "tracked"
    }
]
```

### Running Tests
```powershell
# Unit tests (mocked)
python -m pytest tests/test_bytetrack.py::TestTrackingUtils -v
python -m pytest tests/test_bytetrack.py::TestByteTrackManager -v

# Integration tests (actual tracker)
python -m pytest tests/test_bytetrack.py::TestByteTrackManagerIntegration -v

# All tests
python -m pytest tests/test_bytetrack.py -v
```

### Running Demo Script
```powershell
# Basic usage
python scripts/test_bytetrack.py --source data/videos/test_sample.mp4

# With custom parameters
python scripts/test_bytetrack.py --source data/videos/test_sample.mp4 --confidence 0.4 --imgsz 640 --device cpu --output output/annotated_videos

# With display
python scripts/test_bytetrack.py --source data/videos/test_sample.mp4 --display

# Limit frames
python scripts/test_bytetrack.py --source data/videos/test_sample.mp4 --max-frames 100
```

### Configuration Integration
The tracker integrates with:
- `configs/model.yaml` - Tracking settings (tracker_type, track_thresh, track_buffer, match_thresh, frame_rate, min_box_area)

### Test Results
- TrackingUtils tests: 9/9 passed
- ByteTrackManager unit tests: 7/7 passed
- ByteTrackManager integration tests: 6/6 passed (using SimpleIOUTracker fallback)
- Total: 22/22 tests passed

### Performance Metrics (Synthetic Video Test)
**Note**: The synthetic video contains no actual people, so 0 detections and 0 tracks is correct behavior.

- Tracker: SimpleIOUTracker (fallback - BYTETracker not available in supervision 0.31.3)
- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Processed frames: 30
- Total detections: 0 (expected - synthetic video has no people)
- Total tracks created: 0
- Average detection latency: 1212.39ms
- Average tracking latency: 0.03ms
- Total pipeline latency: 1212.42ms
- Processing FPS: 0.82

### Real Video Validation
**Important**: To validate tracking on actual human-containing video:

1. Download a legally usable test video with people (e.g., from Pexels or Pixabay)
2. Place in `data/videos/` directory
3. Run tracking:
```powershell
python scripts/test_bytetrack.py --source data/videos/your_video.mp4 --output output/annotated_videos
```
4. Check:
   - People are detected and tracked
   - Track IDs remain consistent across frames
   - Track IDs change appropriately for new people
   - Annotated video shows track IDs
   - Tracking report is saved

### Limitations
- Supervision 0.31.3 does not include BYTETracker (using SimpleIOUTracker fallback)
- SimpleIOUTracker is less sophisticated than ByteTrack (no low-confidence track recovery)
- CPU inference is slow (~0.8 FPS with detection + tracking)
- Synthetic video cannot validate tracking behavior
- Requires actual human-containing video for real validation
- No GPU available on current system

### Tracker Information
- **Primary**: ByteTrack (when available in supervision)
- **Fallback**: SimpleIOUTracker (IoU-based matching)
- **Track IDs**: Temporary, reset on tracker reset
- **Track Buffer**: 30 frames (configurable)
- **IoU Threshold**: 0.8 (configurable)
- **Frame Rate**: 30 FPS assumption for track age

---

## Phase 3: YOLO Person Detection Module - Implementation Details

### Files Created
- `ai_engine/__init__.py` - AI engine module initialization
- `ai_engine/detection/__init__.py` - Detection module initialization
- `ai_engine/detection/yolo_detector.py` - PersonDetector class (356 lines)
- `ai_engine/detection/detection_utils.py` - Detection utilities (277 lines)
- `tests/test_yolo_detector.py` - Unit tests for YOLO detector (318 lines)
- `scripts/test_yolo_detection.py` - Demo script for YOLO detection (238 lines)

### PersonDetector Class
- Loads Ultralytics YOLO11n pretrained model
- Automatic model download if not found
- CPU/GPU/MPS device support with auto-detection
- Configurable confidence threshold, IoU threshold, image size
- Detects only COCO class 0 (person)
- Returns standardized detection output with bbox, confidence, class info
- Performance tracking (inference latency, FPS)
- Bounding box drawing and visualization
- Integration with configs/model.yaml

### DetectionUtils Class
- COCO class name mapping (80 classes)
- Bounding box validation and clipping
- Detection filtering by class and confidence
- IoU calculation for bounding boxes
- Bounding box drawing with labels

### Detection Output Format
```python
[
    {
        "class_id": 0,
        "class_name": "person",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.94
    }
]
```

### Running Tests
```powershell
# Unit tests (mocked)
python -m pytest tests/test_yolo_detector.py::TestPersonDetector -v
python -m pytest tests/test_yolo_detector.py::TestDetectionUtils -v

# Integration tests (actual model inference)
python -m pytest tests/test_yolo_detector.py::TestPersonDetectorIntegration -v
```

### Running Demo Script
```powershell
# Basic usage
python scripts/test_yolo_detection.py --source data/videos/test_sample.mp4

# With custom parameters
python scripts/test_yolo_detection.py --source data/videos/test_sample.mp4 --confidence 0.4 --imgsz 640 --device cpu --output output/annotated_videos

# With display
python scripts/test_yolo_detection.py --source data/videos/test_sample.mp4 --display

# Limit frames
python scripts/test_yolo_detection.py --source data/videos/test_sample.mp4 --max-frames 100
```

### Configuration Integration
The YOLO detector integrates with:
- `configs/model.yaml` - YOLO settings (model_name, confidence_threshold, iou_threshold, device, etc.)
- `configs/app.yaml` - Path configurations

### Test Results
- PersonDetector unit tests: 11/11 passed
- DetectionUtils tests: 12/12 passed
- Integration tests (real model): 6/6 passed
- Total: 29/29 tests passed

### Performance Metrics (Synthetic Video Test)
**Note**: The synthetic video (moving_box pattern) contains no actual people, so 0 detections is correct behavior.

- Model: yolo11n.pt
- Device: CPU
- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Processed frames: 30
- Total detections: 0 (expected - synthetic video has no people)
- Average inference latency: 448.78ms
- Median inference latency: 170.27ms
- Min latency: 148.87ms
- Max latency: 8510.93ms (first frame includes model warmup)
- Processing FPS: 2.17

### Real Video Validation
**Important**: To validate person detection on actual human-containing video:

1. Download a legally usable test video with people:
   - Example: Pexels or Pixabay free stock footage
   - Search for "people walking" or "crowd"
   - Ensure video is in MP4 format

2. Place video in `data/videos/` directory

3. Run detection:
```powershell
python scripts/test_yolo_detection.py --source data/videos/your_video.mp4 --output output/annotated_videos
```

4. Check:
   - People are detected (non-zero detection count)
   - Bounding boxes are drawn correctly
   - Confidence scores are reasonable (> 0.5)
   - Annotated video is saved

### Limitations
- CPU inference is slow (~2 FPS for 640x480 resolution)
- First frame has high latency due to model warmup
- Synthetic video cannot be used for person detection validation
- Requires actual human-containing video for real validation
- No GPU available on current system (CPU-only inference)

### Model Information
- **Architecture**: YOLO11n (Nano version of YOLO11)
- **Pretrained on**: COCO dataset
- **Classes**: 80 COCO classes (filtered to person only)
- **Model size**: ~6MB (yolo11n.pt)
- **Input size**: Configurable (default 640x640)
- **Inference**: CPU-compatible, GPU-supported

---

## Phase 2: Video Processing Module - Implementation Details

### Files Created
- `video_processing/__init__.py` - Module initialization
- `video_processing/video_reader.py` - VideoReader class for video I/O
- `video_processing/frame_processor.py` - FrameProcessor class for frame operations
- `video_processing/video_utils.py` - VideoUtils utility functions
- `tests/test_video_reader.py` - Unit tests for video reader and utils
- `tests/test_frame_processor.py` - Unit tests for frame processor
- `scripts/test_video_processing.py` - Demo script for testing

### VideoReader Class
- Opens MP4, AVI, MOV, MKV, WEBM video formats
- Extracts frames sequentially with metadata
- Returns frame number and timestamp
- Reads video FPS, width, height, duration, total frames
- Handles invalid paths and corrupted videos
- Supports context manager usage
- Provides frames generator with optional skipping

### FrameProcessor Class
- Configurable frame resizing with aspect ratio preservation
- Frame validation (None check, type check, shape check)
- Frame skipping logic based on configuration
- Frame preprocessing (normalization, color space conversion)
- Timestamp calculation
- Frame information extraction

### VideoUtils Class
- Synthetic video generation for testing
- Video information retrieval without full opening
- Video validation
- Frame saving as images
- Aspect ratio calculation
- Duration formatting

### Running Tests
```powershell
# Run video reader tests
python -m pytest tests/test_video_reader.py -v

# Run frame processor tests
python -m pytest tests/test_frame_processor.py -v

# Run all video processing tests
python -m pytest tests/test_video_reader.py tests/test_frame_processor.py -v
```

### Running Demo Script
```powershell
# Basic usage
python scripts/test_video_processing.py --source data/videos/test_sample.mp4

# With resizing and frame skipping
python scripts/test_video_processing.py --source data/videos/test_sample.mp4 --resize-width 320 --resize-height 240 --frame-skip 2

# With display (shows window)
python scripts/test_video_processing.py --source data/videos/test_sample.mp4 --display

# Limit frames
python scripts/test_video_processing.py --source data/videos/test_sample.mp4 --max-frames 100
```

### Configuration Integration
The video processing modules integrate with:
- `configs/model.yaml` - Performance settings (resize_width, resize_height, frame_skip)
- `configs/app.yaml` - Path configurations

### Test Results
- VideoReader tests: 12/12 passed
- FrameProcessor tests: 25/25 passed
- VideoUtils tests: 6/6 passed
- Total: 43/43 tests passed

### Performance Metrics (Synthetic Video)
- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Processing: Resize to 320x240, skip 2 frames
- Result: 214.52 FPS processing speed (CPU)

- Total: 43/43 tests passed

### Performance Metrics (Synthetic Video)
- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Processing: Resize to 320x240, skip 2 frames
- Result: 214.52 FPS processing speed (CPU)


---

## Phase 6: Pose Estimation Module - Implementation Details

### Files Created
- `ai_engine/pose/__init__.py` - Pose module initialization and exports
- `ai_engine/pose/pose_estimator.py` - PoseEstimator class for YOLO11n-pose inference
- `ai_engine/pose/pose_utils.py` - PoseUtils class (COCO keypoints, skeleton, association, drawing)
- `ai_engine/pose/pose_analyzer.py` - PoseAnalyzer class for geometric joint-angle analysis
- `tests/test_pose_estimator.py` - Unit tests (58 mocked + 5 smoke tests = 63 total)
- `scripts/test_pose.py` - Demo script for pose estimation

### Files Modified
- `ai_engine/__init__.py` - Added PoseEstimator export
- `configs/model.yaml` - Extended pose section with image_size, iou_threshold, association_iou_threshold, skeleton_drawing

### Architecture

The pose module follows the same patterns as detection and tracking:
- Config-driven via `configs/model.yaml` `pose:` section
- Override via `config` dict parameter
- Performance tracking (inference latency)
- Drawing/visualization methods
- CPU inference by default

**PoseEstimator** loads YOLO11n-pose.pt via Ultralytics and extracts:
- Person bounding boxes (x1, y1, x2, y2)
- COCO 17 keypoints with x, y, confidence per keypoint
- Pose confidence (box detection confidence)

**PoseUtils** provides:
- COCO 17 keypoint names and skeleton connections
- `keypoints_to_dict()` - structured keypoint conversion (null for low-confidence)
- `associate_poses_to_tracks()` - greedy one-to-one IoU matching
- `draw_keypoints()` and `draw_skeleton()` - visualization helpers

**PoseAnalyzer** calculates geometric angles:
- Left/right elbow, knee, hip, shoulder angles (law of cosines)
- Body orientation estimate (front, left_side, right_side, back, unknown)
- Keypoint visibility statistics
- All calculations return None for missing keypoints

### Model Used
- **Architecture**: YOLO11n-pose (Nano pose model)
- **Pretrained on**: COCO dataset
- **Keypoints**: 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Model size**: ~6MB (yolo11n-pose.pt)
- **Input size**: Configurable (default 640, independent of performance.resize_width)
- **Inference**: CPU-compatible, GPU-supported
- **kpt_shape**: [17, 3] — 17 keypoints with (x, y, confidence)

### COCO 17 Keypoints (in model output order)
```
0: nose              9: left_wrist
1: left_eye          10: right_wrist
2: right_eye         11: left_hip
3: left_ear          12: right_hip
4: right_ear         13: left_knee
5: left_shoulder     14: right_knee
6: right_shoulder    15: left_ankle
7: left_elbow        16: right_ankle
8: right_elbow
```

### Tracking Association

Pose-to-track association uses deterministic greedy IoU matching:
1. Build IoU matrix between all pose bboxes and tracking bboxes
2. Sort candidates by IoU descending
3. Assign greedily — each pose gets at most one track_id, each track_id assigned to at most one pose
4. Association statuses:
   - "matched": IoU >= threshold, one-to-one assignment
   - "unmatched": no track has IoU >= threshold → track_id = None
   - "uncertain": best candidate already assigned to another pose → track_id = None

Default association IoU threshold: 0.5 (configurable in model.yaml)

### Output JSON Structure
```python
{
    "camera_id": "cam_001",
    "frame_number": 100,
    "poses": [
        {
            "track_id": 1,
            "association_status": "matched",
            "pose_confidence": 0.91,
            "bbox": [100, 50, 300, 450],
            "keypoints": {
                "nose": {"x": 200, "y": 70, "confidence": 0.95},
                "left_shoulder": {"x": 160, "y": 150, "confidence": 0.92},
                "left_knee": null,  # below threshold
                ...
            }
        }
    ]
}
```

### Configuration
The pose module reads from `configs/model.yaml` `pose:` section:
```yaml
pose:
  model_name: "yolo11n-pose.pt"
  confidence_threshold: 0.5
  keypoint_threshold: 0.5
  device: "cpu"
  image_size: 640
  iou_threshold: 0.45
  association_iou_threshold: 0.5
  skeleton_drawing:
    keypoint_color: [0, 255, 255]
    skeleton_color: [0, 255, 0]
    bbox_color: [0, 200, 0]
    keypoint_radius: 4
    line_thickness: 2
```

### Running Tests
```powershell
# Mocked unit tests (no model download required)
python -m pytest tests/test_pose_estimator.py -v -k "not SmokeTest"

# Model smoke tests (actual YOLO11n-pose loading and CPU inference)
python -m pytest tests/test_pose_estimator.py::TestPoseModelSmokeTest -v

# All pose tests
python -m pytest tests/test_pose_estimator.py -v
```

### Running Demo Script
```powershell
# Basic usage
python scripts/test_pose.py --source data/videos/test_sample.mp4

# With output and frame limit
python scripts/test_pose.py --source data/videos/test_sample.mp4 --output output/test_pose --max-frames 30

# With display
python scripts/test_pose.py --source data/videos/test_sample.mp4 --display

# Custom parameters
python scripts/test_pose.py --source data/videos/your_video.mp4 --confidence 0.5 --keypoint-threshold 0.5 --imgsz 640 --device cpu
```

### Test Results
- PoseUtils tests: 12/12 passed
- PoseEstimator init tests: 6/6 passed
- PoseEstimator performance tests: 2/2 passed
- PoseEstimator estimate tests: 11/11 passed
- PoseEstimator drawing tests: 3/3 passed
- Tracking association tests: 7/7 passed
- JSON serialization tests: 3/3 passed
- PoseAnalyzer tests: 7/7 passed
- Estimate with tracking tests: 4/4 passed
- Model smoke tests: 5/5 passed (actual YOLO11n-pose CPU inference)
- Total pose tests: 63/63 passed
- Full regression: 205/205 passed (142 existing + 63 new)

### Performance Metrics (Synthetic Video Test)
**Note**: Synthetic video (test_sample.mp4) contains no actual people, so 0 poses is correct behavior.

- Model: yolo11n-pose.pt
- Device: CPU
- Video: 640x480, 30 FPS, 3 seconds (90 frames)
- Model load time: 0.34s
- Processed frames: 10
- Total poses: 0 (expected — synthetic video has no people)
- Average inference latency: 1074.43ms (first frame warmup: 8238ms)
- Median inference latency: 224.16ms
- Min inference latency: 160.93ms
- Max inference latency: 8238.13ms (first frame warmup)
- Effective FPS: 0.92

### Real Video Validation
**Important**: To validate pose estimation on actual human-containing video:

1. Download a legally usable test video with people (e.g., from Pexels or Pixabay)
2. Place in `data/videos/` directory
3. Run pose estimation:
```powershell
python scripts/test_pose.py --source data/videos/your_video.mp4 --output output/test_pose
```
4. Check:
   - People are detected with bounding boxes
   - Skeleton connections are drawn between visible keypoints
   - Keypoint circles are drawn for confident keypoints
   - Pose report JSON is saved with latency and FPS metrics
   - Pose results JSON is saved per-frame with keypoints and analysis
   - Annotated video shows skeleton overlays

### Known Limitations
- CPU inference is slow (~0.9 FPS for pose-only, median ~224ms per frame after warmup)
- First frame has high latency due to model warmup (~8s)
- 2D keypoint projections may not reflect true 3D joint angles
- Body orientation is a rough estimate and may be inaccurate
- Requires actual human-containing video for real validation
- No GPU available on current system (CPU-only inference)
- Pose model detects persons independently — does not use PersonDetector
- Pose-to-track association requires ByteTrackManager tracking output as input
- Joint angles are None when any of the three required keypoints are missing
