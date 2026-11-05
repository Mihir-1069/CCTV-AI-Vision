"""
Unit tests for YOLO Person Detector module.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.detection.yolo_detector import PersonDetector
from ai_engine.detection.detection_utils import DetectionUtils


class TestPersonDetector:
    """Test cases for PersonDetector class."""
    
    @pytest.fixture
    def detector_config(self):
        """Configuration for detector."""
        return {
            "model_name": "yolo11n.pt",
            "device": "cpu",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "image_size": 640,
            "half_precision": False,
            "target_classes": [0],
            "max_detections": 100,
            "verbose": False
        }
    
    @pytest.fixture
    def detector(self, detector_config):
        """Create PersonDetector instance."""
        return PersonDetector(config=detector_config)
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector_config):
        """Test detector initialization."""
        detector = PersonDetector(config=detector_config)
        
        assert detector.model_name == "yolo11n.pt"
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45
        assert detector.image_size == 640
        assert detector.half_precision is False
        assert detector.target_classes == [0]
        assert detector.max_detections == 100
        assert detector.verbose is False
    
    def test_detector_initialization_with_config_file(self):
        """Test detector initialization with config file."""
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
        detector = PersonDetector(config_path=str(config_path))
        
        assert detector.model_name == "yolo11n.pt"
        assert detector.device == "cpu"
    
    def test_get_device(self, detector):
        """Test getting device."""
        device = detector.get_device()
        assert device == "cpu"
    
    def test_get_model_info(self, detector):
        """Test getting model information."""
        info = detector.get_model_info()
        
        assert info["model_name"] == "yolo11n.pt"
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.5
        assert info["iou_threshold"] == 0.45
        assert info["image_size"] == 640
        assert info["half_precision"] is False
        assert info["target_classes"] == [0]
        assert info["max_detections"] == 100
        assert info["is_loaded"] is False
    
    def test_detect_without_model(self, detector, sample_frame):
        """Test detection without loading model."""
        detections = detector.detect(sample_frame)
        
        assert detections == []
    
    def test_detect_invalid_frame_none(self, detector):
        """Test detection with None frame."""
        detector.model = Mock()  # Mock model to bypass check
        detections = detector.detect(None)
        
        assert detections == []
    
    def test_detect_invalid_frame_not_array(self, detector):
        """Test detection with non-array frame."""
        detector.model = Mock()
        detections = detector.detect("not_an_array")
        
        assert detections == []
    
    def test_detect_invalid_frame_shape(self, detector):
        """Test detection with invalid frame shape."""
        detector.model = Mock()
        invalid_frame = np.array([1, 2, 3])
        detections = detector.detect(invalid_frame)
        
        assert detections == []
    
    def test_get_performance_stats_empty(self, detector):
        """Test performance stats with no inferences."""
        stats = detector.get_performance_stats()
        
        assert stats["inference_count"] == 0
        assert stats["avg_latency_ms"] is None
        assert stats["median_latency_ms"] is None
        assert stats["min_latency_ms"] is None
        assert stats["max_latency_ms"] is None
    
    def test_reset_performance_stats(self, detector):
        """Test resetting performance stats."""
        detector._inference_times = [0.1, 0.2, 0.3]
        detector.reset_performance_stats()
        
        assert detector._inference_times == []
    
    def test_detector_repr(self, detector):
        """Test detector string representation."""
        repr_str = repr(detector)
        
        assert "PersonDetector" in repr_str
        assert "yolo11n.pt" in repr_str
        assert "cpu" in repr_str
        assert "not loaded" in repr_str


class TestDetectionUtils:
    """Test cases for DetectionUtils class."""
    
    def test_get_class_name(self):
        """Test getting class name from ID."""
        assert DetectionUtils.get_class_name(0) == "person"
        assert DetectionUtils.get_class_name(1) == "bicycle"
        assert DetectionUtils.get_class_name(99) == "unknown"
    
    def test_validate_bbox_valid(self):
        """Test validating valid bounding box."""
        bbox = [10, 10, 100, 100]
        assert DetectionUtils.validate_bbox(bbox, 640, 480) is True
    
    def test_validate_bbox_invalid_length(self):
        """Test validating bbox with invalid length."""
        bbox = [10, 10, 100]
        assert DetectionUtils.validate_bbox(bbox, 640, 480) is False
    
    def test_validate_bbox_out_of_bounds(self):
        """Test validating bbox out of bounds."""
        bbox = [10, 10, 1000, 1000]
        assert DetectionUtils.validate_bbox(bbox, 640, 480) is False
    
    def test_validate_bbox_invalid_coordinates(self):
        """Test validating bbox with invalid coordinates."""
        bbox = [100, 100, 50, 50]  # x2 < x1
        assert DetectionUtils.validate_bbox(bbox, 640, 480) is False
    
    def test_clip_bbox(self):
        """Test clipping bounding box."""
        bbox = [10, 10, 1000, 1000]
        clipped = DetectionUtils.clip_bbox(bbox, 640, 480)
        
        assert clipped == [10, 10, 640, 480]
    
    def test_clip_bbox_negative(self):
        """Test clipping bbox with negative coordinates."""
        bbox = [-10, -10, 100, 100]
        clipped = DetectionUtils.clip_bbox(bbox, 640, 480)
        
        assert clipped == [0, 0, 100, 100]
    
    def test_filter_detections_by_class(self):
        """Test filtering detections by class."""
        detections = [
            {"class_id": 0, "class_name": "person", "bbox": [10, 10, 100, 100], "confidence": 0.9},
            {"class_id": 1, "class_name": "bicycle", "bbox": [20, 20, 120, 120], "confidence": 0.8},
            {"class_id": 0, "class_name": "person", "bbox": [30, 30, 130, 130], "confidence": 0.7},
        ]
        
        filtered = DetectionUtils.filter_detections_by_class(detections, [0])
        
        assert len(filtered) == 2
        assert all(det["class_id"] == 0 for det in filtered)
    
    def test_filter_detections_by_confidence(self):
        """Test filtering detections by confidence."""
        detections = [
            {"class_id": 0, "confidence": 0.9},
            {"class_id": 0, "confidence": 0.4},
            {"class_id": 0, "confidence": 0.7},
        ]
        
        filtered = DetectionUtils.filter_detections_by_confidence(detections, 0.5)
        
        assert len(filtered) == 2
        assert all(det["confidence"] >= 0.5 for det in filtered)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        
        iou = DetectionUtils.calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0
    
    def test_calculate_iou_full_overlap(self):
        """Test IoU calculation with full overlap."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [0, 0, 10, 10]
        
        iou = DetectionUtils.calculate_iou(bbox1, bbox2)
        
        assert iou == 1.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        
        iou = DetectionUtils.calculate_iou(bbox1, bbox2)
        
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 ≈ 0.143
        assert 0.14 < iou < 0.15


class TestPersonDetectorIntegration:
    """Integration tests with actual model inference."""
    
    @pytest.fixture
    def real_detector(self):
        """Create detector with actual model."""
        detector = PersonDetector(config={"device": "cpu", "verbose": False})
        success = detector.load_model()
        if not success:
            pytest.skip("Could not load YOLO model")
        return detector
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_load_model(self, real_detector):
        """Test loading actual model."""
        assert real_detector.model is not None
        assert real_detector.get_model_info()["is_loaded"] is True
    
    def test_detect_real_frame(self, real_detector, sample_frame):
        """Test detection on real frame."""
        detections = real_detector.detect(sample_frame)
        
        # Should return a list (may be empty for random noise)
        assert isinstance(detections, list)
    
    def test_detection_output_schema(self, real_detector, sample_frame):
        """Test detection output schema."""
        detections = real_detector.detect(sample_frame)
        
        for det in detections:
            assert "class_id" in det
            assert "class_name" in det
            assert "bbox" in det
            assert "confidence" in det
            assert isinstance(det["class_id"], int)
            assert isinstance(det["class_name"], str)
            assert isinstance(det["bbox"], list)
            assert len(det["bbox"]) == 4
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0
    
    def test_draw_detections(self, real_detector, sample_frame):
        """Test drawing detections on frame."""
        detections = real_detector.detect(sample_frame)
        annotated = real_detector.draw_detections(sample_frame, detections)
        
        # Original frame should not be modified
        assert not np.array_equal(sample_frame, annotated) if detections else True
        assert annotated.shape == sample_frame.shape
    
    def test_confidence_threshold_filtering(self, real_detector, sample_frame):
        """Test confidence threshold filtering."""
        # Test with high threshold
        real_detector.confidence_threshold = 0.99
        detections_high = real_detector.detect(sample_frame)
        
        # Test with low threshold
        real_detector.confidence_threshold = 0.1
        detections_low = real_detector.detect(sample_frame)
        
        # Low threshold should have more or equal detections
        assert len(detections_low) >= len(detections_high)
    
    def test_performance_stats_tracking(self, real_detector, sample_frame):
        """Test performance statistics tracking."""
        real_detector.reset_performance_stats()
        
        # Run multiple detections
        for _ in range(5):
            real_detector.detect(sample_frame)
        
        stats = real_detector.get_performance_stats()
        
        assert stats["inference_count"] == 5
        assert stats["avg_latency_ms"] is not None
        assert stats["median_latency_ms"] is not None
        assert stats["min_latency_ms"] is not None
        assert stats["max_latency_ms"] is not None
        assert stats["avg_latency_ms"] > 0
