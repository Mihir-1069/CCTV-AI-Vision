"""
Unit tests for Crowd Analytics module.
"""

import pytest
import numpy as np

from ai_engine.crowd.crowd_analytics import CrowdAnalytics, CrowdMetrics
from ai_engine.analytics.zone_manager import ZoneManager


class TestCrowdMetrics:
    """Test cases for CrowdMetrics class."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create a sample CrowdMetrics."""
        return CrowdMetrics(
            timestamp="2026-11-05T10:00:00Z",
            frame_number=100,
            camera_id="cam_001",
            total_count=5,
            density=1.5,
            avg_velocity=2.0,
            movement_direction=45.0,
            congestion_level="MEDIUM",
            zone_metrics={"zone_1": {"count": 3, "density": 2.0}}
        )
    
    def test_to_dict(self, sample_metrics):
        """Test JSON serialization."""
        result = sample_metrics.to_dict()
        assert isinstance(result, dict)
        assert result["camera_id"] == "cam_001"
        assert result["total_count"] == 5
        assert result["congestion_level"] == "MEDIUM"
        assert isinstance(result["zone_metrics"], dict)


class TestCrowdAnalytics:
    """Test cases for CrowdAnalytics class."""
    
    @pytest.fixture
    def zone_manager(self):
        """Create a ZoneManager with a zone."""
        zones_config = {
            "main_area": {
                "id": "zone_001",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]
            }
        }
        return ZoneManager(camera_id="cam_001", zones_config=zones_config)
    
    @pytest.fixture
    def crowd_analytics(self, zone_manager):
        """Create a CrowdAnalytics instance."""
        return CrowdAnalytics(
            camera_id="cam_001",
            zone_manager=zone_manager,
            frame_width=1920,
            frame_height=1080
        )
    
    @pytest.fixture
    def sample_tracking(self):
        """Create sample tracking output."""
        return [
            {"track_id": 1, "bbox": [10, 10, 50, 50]},
            {"track_id": 2, "bbox": [100, 100, 150, 150]},
            {"track_id": 3, "bbox": [200, 200, 250, 250]},
        ]
    
    def test_initialization(self, crowd_analytics):
        """Test CrowdAnalytics initialization."""
        assert crowd_analytics.camera_id == "cam_001"
        assert crowd_analytics.frame_width == 1920
        assert crowd_analytics.frame_height == 1080
        assert len(crowd_analytics._metrics_history) == 0
    
    def test_update_basic_metrics(self, crowd_analytics, sample_tracking):
        """Test basic metrics calculation."""
        metrics = crowd_analytics.update(sample_tracking, frame_number=100)
        
        assert metrics.total_count == 3
        assert metrics.density > 0
        assert metrics.camera_id == "cam_001"
        assert metrics.frame_number == 100
    
    def test_density_calculation(self, crowd_analytics):
        """Test density calculation."""
        tracking = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        metrics = crowd_analytics.update(tracking, frame_number=100)
        
        frame_area = 1920 * 1080
        expected_density = (1 / frame_area) * 1000.0
        assert abs(metrics.density - expected_density) < 0.0001
    
    def test_congestion_levels(self, crowd_analytics):
        """Test congestion level determination."""
        # Low congestion
        crowd_analytics.density_threshold_low = 0.5
        crowd_analytics.density_threshold_high = 2.0
        
        tracking = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        metrics = crowd_analytics.update(tracking, frame_number=100)
        assert metrics.congestion_level == "LOW"
        
        # High congestion (simulated by adjusting thresholds)
        crowd_analytics.density_threshold_low = 0.0
        crowd_analytics.density_threshold_high = 0.0001
        metrics = crowd_analytics.update(tracking, frame_number=101)
        assert metrics.congestion_level == "HIGH"
    
    def test_velocity_calculation(self, crowd_analytics):
        """Test velocity calculation across frames."""
        tracking1 = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        crowd_analytics.update(tracking1, frame_number=100)
        
        tracking2 = [{"track_id": 1, "bbox": [20, 20, 60, 60]}]
        metrics = crowd_analytics.update(tracking2, frame_number=101)
        
        # Velocity should be non-zero (center moved from (30,30) to (40,40))
        assert metrics.avg_velocity > 0
    
    def test_zone_metrics(self, crowd_analytics, sample_tracking):
        """Test zone-based metrics calculation."""
        # Update zone manager with tracking data
        crowd_analytics.zone_manager.update_zones(sample_tracking)
        
        metrics = crowd_analytics.update(sample_tracking, frame_number=100)
        
        assert "main_area" in metrics.zone_metrics
        assert "zone_id" in metrics.zone_metrics["main_area"]
        assert "count" in metrics.zone_metrics["main_area"]
        assert "density" in metrics.zone_metrics["main_area"]
    
    def test_smoothed_metrics(self, crowd_analytics, sample_tracking):
        """Test temporal smoothing."""
        # Add multiple frames
        for i in range(10):
            crowd_analytics.update(sample_tracking, frame_number=i)
        
        smoothed = crowd_analytics.get_smoothed_metrics(window_size=5)
        assert smoothed is not None
        assert smoothed.total_count == 3
    
    def test_smoothed_metrics_insufficient_history(self, crowd_analytics, sample_tracking):
        """Test smoothing with insufficient history."""
        crowd_analytics.update(sample_tracking, frame_number=100)
        smoothed = crowd_analytics.get_smoothed_metrics(window_size=5)
        assert smoothed is None
    
    def test_congestion_history(self, crowd_analytics, sample_tracking):
        """Test congestion history tracking."""
        crowd_analytics.update(sample_tracking, frame_number=100)
        crowd_analytics.update(sample_tracking, frame_number=101)
        
        history = crowd_analytics.get_congestion_history()
        assert len(history) == 2
        assert "timestamp" in history[0]
        assert "congestion_level" in history[0]
    
    def test_reset(self, crowd_analytics, sample_tracking):
        """Test resetting the analytics."""
        crowd_analytics.update(sample_tracking, frame_number=100)
        assert len(crowd_analytics._metrics_history) > 0
        
        crowd_analytics.reset()
        assert len(crowd_analytics._metrics_history) == 0
        assert len(crowd_analytics._track_velocity_history) == 0
    
    def test_without_zone_manager(self):
        """Test analytics without zone manager."""
        analytics = CrowdAnalytics(
            camera_id="cam_001",
            zone_manager=None,
            frame_width=1920,
            frame_height=1080
        )
        
        tracking = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        metrics = analytics.update(tracking, frame_number=100)
        
        assert metrics.total_count == 1
        assert len(metrics.zone_metrics) == 0
    
    def test_invalid_tracking_data(self, crowd_analytics):
        """Test handling of invalid tracking data."""
        invalid_tracking = [
            {"track_id": -1, "bbox": [10, 10, 50, 50]},  # Invalid track ID
            {"track_id": 1, "bbox": []},  # Invalid bbox
            {"track_id": 2, "bbox": [10, 10, 50]},  # Incomplete bbox
        ]
        
        metrics = crowd_analytics.update(invalid_tracking, frame_number=100)
        assert metrics.total_count == 0
    
    def test_polygon_area_calculation(self, crowd_analytics):
        """Test polygon area calculation."""
        # Square 100x100
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        area = crowd_analytics._calculate_polygon_area(polygon)
        assert area == 10000.0
        
        # Triangle
        triangle = [[0, 0], [100, 0], [50, 100]]
        area = crowd_analytics._calculate_polygon_area(triangle)
        assert area == 5000.0
