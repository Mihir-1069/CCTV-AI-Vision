"""
Unit tests for ByteTrack tracking module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.tracking.bytetrack_manager import ByteTrackManager
from ai_engine.tracking.tracking_utils import TrackingUtils


class TestTrackingUtils:
    """Test cases for TrackingUtils class."""
    
    def test_convert_detections_to_supervision_empty(self):
        """Test converting empty detections."""
        detections = []
        result = TrackingUtils.convert_detections_to_supervision(detections)
        
        assert result.shape == (0, 6)
    
    def test_convert_detections_to_supervision(self):
        """Test converting detections to supervision format."""
        detections = [
            {"bbox": [10, 10, 100, 100], "confidence": 0.9, "class_id": 0},
            {"bbox": [20, 20, 120, 120], "confidence": 0.8, "class_id": 0},
        ]
        
        result = TrackingUtils.convert_detections_to_supervision(detections)
        
        assert result.shape == (2, 6)
        assert result[0, 0] == 10  # x1
        assert result[0, 1] == 10  # y1
        assert result[0, 2] == 100  # x2
        assert result[0, 3] == 100  # y2
        assert result[0, 4] == 0.9  # confidence
        assert result[0, 5] == 0  # class_id
    
    def test_convert_supervision_to_tracking_output_empty(self):
        """Test converting empty supervision results."""
        detections = np.empty((0, 6))
        track_ids = np.array([])
        
        result = TrackingUtils.convert_supervision_to_tracking_output(detections, track_ids)
        
        assert result == []
    
    def test_convert_supervision_to_tracking_output(self):
        """Test converting supervision results to tracking output."""
        detections = np.array([
            [10, 10, 100, 100, 0.9, 0],
            [20, 20, 120, 120, 0.8, 0],
        ])
        track_ids = np.array([1, 2])
        
        result = TrackingUtils.convert_supervision_to_tracking_output(detections, track_ids)
        
        assert len(result) == 2
        assert result[0]["track_id"] == 1
        assert result[0]["class_name"] == "person"
        assert result[0]["bbox"] == [10, 10, 100, 100]
        assert result[0]["confidence"] == 0.9
        assert result[0]["track_state"] == "tracked"
    
    def test_draw_track_id(self):
        """Test drawing track ID on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [10, 10, 100, 100]
        track_id = 1
        
        result = TrackingUtils.draw_track_id(frame, bbox, track_id)
        
        assert result.shape == frame.shape
        # Original frame should not be modified
        assert not np.array_equal(frame, result)
    
    def test_draw_tracking_info(self):
        """Test drawing tracking info overlay."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = TrackingUtils.draw_tracking_info(
            frame, active_tracks=5, frame_number=100,
            detection_latency=50.0, tracking_latency=10.0
        )
        
        assert result.shape == frame.shape
        assert not np.array_equal(frame, result)
    
    def test_calculate_track_age(self):
        """Test calculating track age."""
        track_history = [
            {"frame_number": 0},
            {"frame_number": 10},
            {"frame_number": 20},
        ]
        
        age = TrackingUtils.calculate_track_age(track_history, current_frame=30)
        
        assert age == 30
    
    def test_calculate_track_age_empty(self):
        """Test calculating track age with empty history."""
        age = TrackingUtils.calculate_track_age([], current_frame=30)
        
        assert age == 0
    
    def test_filter_tracks_by_age(self):
        """Test filtering tracks by age."""
        tracks = [
            {"age": 5},
            {"age": 15},
            {"age": 25},
            {"age": 35},
        ]
        
        filtered = TrackingUtils.filter_tracks_by_age(tracks, min_age=10, max_age=30)
        
        assert len(filtered) == 2
        assert filtered[0]["age"] == 15
        assert filtered[1]["age"] == 25


class TestByteTrackManager:
    """Test cases for ByteTrackManager class."""
    
    @pytest.fixture
    def tracker_config(self):
        """Configuration for tracker."""
        return {
            "tracker_type": "bytetrack",
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
            "min_box_area": 10,
            "frame_rate": 30
        }
    
    @pytest.fixture
    def tracker(self, tracker_config):
        """Create ByteTrackManager instance."""
        return ByteTrackManager(config=tracker_config)
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                "class_id": 0,
                "class_name": "person",
                "bbox": [10, 10, 100, 100],
                "confidence": 0.9
            },
            {
                "class_id": 0,
                "class_name": "person",
                "bbox": [20, 20, 120, 120],
                "confidence": 0.8
            }
        ]
    
    def test_tracker_initialization(self, tracker_config):
        """Test tracker initialization."""
        tracker = ByteTrackManager(config=tracker_config)
        
        assert tracker.tracker_type == "bytetrack"
        assert tracker.track_thresh == 0.5
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.8
        assert tracker.fuse_score is True
        assert tracker.min_box_area == 10
        assert tracker.frame_rate == 30
    
    def test_tracker_initialization_with_config_file(self):
        """Test tracker initialization with config file."""
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
        tracker = ByteTrackManager(config_path=str(config_path))
        
        assert tracker.tracker_type == "bytetrack"
    
    def test_get_tracking_stats_empty(self, tracker):
        """Test tracking stats with no updates."""
        stats = tracker.get_tracking_stats()
        
        assert stats["tracking_count"] == 0
        assert stats["avg_tracking_latency_ms"] is None
        assert stats["median_tracking_latency_ms"] is None
        assert stats["min_tracking_latency_ms"] is None
        assert stats["max_tracking_latency_ms"] is None
        assert stats["total_tracks_created"] == 0
        assert stats["active_tracks"] == 0
    
    def test_get_active_tracks_empty(self, tracker):
        """Test getting active tracks with no updates."""
        active = tracker.get_active_tracks()
        
        assert active == []
    
    def test_get_track_history_empty(self, tracker):
        """Test getting track history for non-existent track."""
        history = tracker.get_track_history(track_id=999)
        
        assert history == []
    
    def test_tracker_reset(self, tracker):
        """Test tracker reset."""
        tracker._frame_number = 100
        tracker._total_tracks_created = 5
        tracker._active_tracks[1] = {"track_id": 1}
        
        tracker.reset()
        
        assert tracker._frame_number == 0
        assert tracker._total_tracks_created == 0
        assert len(tracker._active_tracks) == 0
    
    def test_tracker_repr(self, tracker):
        """Test tracker string representation."""
        repr_str = repr(tracker)
        
        assert "ByteTrackManager" in repr_str
        assert "active_tracks" in repr_str
    
    def test_update_empty_detections(self, tracker):
        """Test update with empty detections."""
        detections = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock tracker to avoid actual tracking
        with patch.object(tracker, 'tracker', None):
            result = tracker.update(detections, frame)
        
        assert result == []
    
    def test_update_without_tracker(self, tracker):
        """Test update when tracker is not initialized."""
        tracker.tracker = None
        detections = [{"bbox": [10, 10, 100, 100], "confidence": 0.9, "class_id": 0}]
        
        result = tracker.update(detections)
        
        assert result == []
    
    def test_draw_tracks_empty(self, tracker):
        """Test drawing tracks with empty tracking output."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking_output = []
        
        result = tracker.draw_tracks(frame, tracking_output)
        
        assert result.shape == frame.shape


class TestByteTrackManagerIntegration:
    """Integration tests with actual ByteTrack (if available)."""
    
    @pytest.fixture
    def real_tracker(self):
        """Create tracker with actual ByteTrack."""
        tracker = ByteTrackManager(config={"verbose": False})
        if tracker.tracker is None:
            pytest.skip("ByteTracker not available")
        return tracker
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                "class_id": 0,
                "class_name": "person",
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9
            }
        ]
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialize_tracker(self, real_tracker):
        """Test initializing actual tracker."""
        assert real_tracker.tracker is not None
    
    def test_update_with_detections(self, real_tracker, sample_detections, sample_frame):
        """Test update with actual detections."""
        result = real_tracker.update(sample_detections, sample_frame)
        
        assert isinstance(result, list)
    
    def test_tracking_output_schema(self, real_tracker, sample_detections, sample_frame):
        """Test tracking output schema."""
        result = real_tracker.update(sample_detections, sample_frame)
        
        for track in result:
            assert "track_id" in track
            assert "class_id" in track
            assert "class_name" in track
            assert "bbox" in track
            assert "confidence" in track
            assert "track_state" in track
            assert isinstance(track["bbox"], list)
            assert len(track["bbox"]) == 4
    
    def test_multiple_updates(self, real_tracker, sample_detections, sample_frame):
        """Test multiple tracker updates."""
        # Update multiple times with similar detections
        for _ in range(5):
            real_tracker.update(sample_detections, sample_frame)
        
        stats = real_tracker.get_tracking_stats()
        
        assert stats["tracking_count"] == 5
        assert stats["avg_tracking_latency_ms"] is not None
    
    def test_track_history(self, real_tracker, sample_detections, sample_frame):
        """Test track history tracking."""
        real_tracker.update(sample_detections, sample_frame)
        
        active_tracks = real_tracker.get_active_tracks()
        
        if len(active_tracks) > 0:
            track_id = active_tracks[0]["track_id"]
            history = real_tracker.get_track_history(track_id)
            
            assert isinstance(history, list)
    
    def test_draw_tracks(self, real_tracker, sample_detections, sample_frame):
        """Test drawing tracks."""
        tracking_output = real_tracker.update(sample_detections, sample_frame)
        
        annotated = real_tracker.draw_tracks(sample_frame, tracking_output)
        
        assert annotated.shape == sample_frame.shape
        # Original frame should not be modified
        assert not np.array_equal(sample_frame, annotated) if tracking_output else True
