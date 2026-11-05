"""
Unit tests for Analytics module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.analytics.analytics_utils import AnalyticsUtils
from ai_engine.analytics.occupancy_manager import OccupancyManager
from ai_engine.analytics.entry_exit_counter import EntryExitCounter
from ai_engine.analytics.zone_manager import ZoneManager


class TestAnalyticsUtils:
    """Test cases for AnalyticsUtils class."""
    
    def test_calculate_bbox_center(self):
        """Test calculating bounding box center."""
        bbox = [0, 0, 100, 100]
        center = AnalyticsUtils.calculate_bbox_center(bbox)
        
        assert center == (50.0, 50.0)
    
    def test_calculate_bbox_center_offset(self):
        """Test calculating bbox center with offset."""
        bbox = [10, 20, 30, 40]
        center = AnalyticsUtils.calculate_bbox_center(bbox)
        
        assert center == (20.0, 30.0)
    
    def test_point_to_line_side_positive(self):
        """Test point to line side calculation (positive)."""
        point = (10, 10)
        line_start = (0, 0)
        line_end = (10, 0)
        
        side = AnalyticsUtils.point_to_line_side(point, line_start, line_end)
        
        assert side > 0
    
    def test_point_to_line_side_negative(self):
        """Test point to line side calculation (negative)."""
        point = (10, -10)
        line_start = (0, 0)
        line_end = (10, 0)
        
        side = AnalyticsUtils.point_to_line_side(point, line_start, line_end)
        
        assert side < 0
    
    def test_point_to_line_side_on_line(self):
        """Test point on line."""
        point = (5, 0)
        line_start = (0, 0)
        line_end = (10, 0)
        
        side = AnalyticsUtils.point_to_line_side(point, line_start, line_end)
        
        assert side == 0
    
    def test_detect_line_crossing_entry(self):
        """Test detecting entry crossing."""
        prev_point = (5, -10)
        curr_point = (5, 10)
        line_start = (0, 0)
        line_end = (10, 0)
        
        crossing = AnalyticsUtils.detect_line_crossing(
            prev_point, curr_point, line_start, line_end
        )
        
        assert crossing == "entry"
    
    def test_detect_line_crossing_exit(self):
        """Test detecting exit crossing."""
        prev_point = (5, 10)
        curr_point = (5, -10)
        line_start = (0, 0)
        line_end = (10, 0)
        
        crossing = AnalyticsUtils.detect_line_crossing(
            prev_point, curr_point, line_start, line_end
        )
        
        assert crossing == "exit"
    
    def test_detect_line_crossing_none(self):
        """Test no crossing detected."""
        prev_point = (5, 10)
        curr_point = (5, 20)
        line_start = (0, 0)
        line_end = (10, 0)
        
        crossing = AnalyticsUtils.detect_line_crossing(
            prev_point, curr_point, line_start, line_end
        )
        
        assert crossing == "none"
    
    def test_point_in_polygon_inside(self):
        """Test point inside polygon."""
        point = (50, 50)
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        inside = AnalyticsUtils.point_in_polygon(point, polygon)
        
        assert inside is True
    
    def test_point_in_polygon_outside(self):
        """Test point outside polygon."""
        point = (150, 150)
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        inside = AnalyticsUtils.point_in_polygon(point, polygon)
        
        assert inside is False
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        dt = datetime(2026, 11, 5, 18, 30, 0)
        timestamp = AnalyticsUtils.format_timestamp(dt)
        
        assert "2026-11-05" in timestamp
        assert "18:30:00" in timestamp
    
    def test_validate_analytics_output_valid(self):
        """Test validating valid analytics output."""
        output = {
            "camera_id": "cam_001",
            "timestamp": "2026-11-05T18:30:00",
            "frame_number": 150,
            "occupancy": {
                "current_count": 5
            }
        }
        
        valid = AnalyticsUtils.validate_analytics_output(output)
        
        assert valid is True
    
    def test_validate_analytics_output_invalid(self):
        """Test validating invalid analytics output."""
        output = {
            "camera_id": "cam_001",
            "timestamp": "2026-11-05T18:30:00"
            # Missing frame_number and occupancy
        }
        
        valid = AnalyticsUtils.validate_analytics_output(output)
        
        assert valid is False
    
    def test_calculate_distance(self):
        """Test calculating distance between points."""
        point1 = (0, 0)
        point2 = (3, 4)
        
        distance = AnalyticsUtils.calculate_distance(point1, point2)
        
        assert distance == 5.0


class TestOccupancyManager:
    """Test cases for OccupancyManager class."""
    
    @pytest.fixture
    def occupancy_manager(self):
        """Create OccupancyManager instance."""
        return OccupancyManager(camera_id="cam_001")
    
    @pytest.fixture
    def sample_tracking_output(self):
        """Create sample tracking output."""
        return [
            {"track_id": 1, "bbox": [10, 10, 50, 50], "confidence": 0.9},
            {"track_id": 2, "bbox": [100, 100, 150, 150], "confidence": 0.8},
            {"track_id": 3, "bbox": [200, 200, 250, 250], "confidence": 0.7},
        ]
    
    def test_initialization(self, occupancy_manager):
        """Test occupancy manager initialization."""
        assert occupancy_manager.camera_id == "cam_001"
        assert occupancy_manager.get_current_count() == 0
    
    def test_update_with_tracking_output(self, occupancy_manager, sample_tracking_output):
        """Test updating occupancy with tracking output."""
        stats = occupancy_manager.update(sample_tracking_output, frame_number=100)
        
        assert occupancy_manager.get_current_count() == 3
        assert stats["occupancy"]["current_count"] == 3
    
    def test_update_with_empty_tracking(self, occupancy_manager):
        """Test updating with empty tracking output."""
        stats = occupancy_manager.update([], frame_number=100)
        
        assert occupancy_manager.get_current_count() == 0
        assert stats["occupancy"]["current_count"] == 0
    
    def test_get_current_stats(self, occupancy_manager, sample_tracking_output):
        """Test getting current stats."""
        occupancy_manager.update(sample_tracking_output, frame_number=100)
        stats = occupancy_manager.get_current_stats()
        
        assert "camera_id" in stats
        assert "timestamp" in stats
        assert "occupancy" in stats
        assert stats["occupancy"]["current_count"] == 3
        assert stats["occupancy"]["max_count"] == 3
        assert stats["occupancy"]["min_count"] == 3
    
    def test_get_active_track_ids(self, occupancy_manager, sample_tracking_output):
        """Test getting active track IDs."""
        occupancy_manager.update(sample_tracking_output, frame_number=100)
        track_ids = occupancy_manager.get_active_track_ids()
        
        assert track_ids == {1, 2, 3}
    
    def test_max_occupancy_tracking(self, occupancy_manager):
        """Test max occupancy tracking."""
        # Update with 3 people
        tracking_3 = [
            {"track_id": 1, "bbox": [10, 10, 50, 50]},
            {"track_id": 2, "bbox": [100, 100, 150, 150]},
            {"track_id": 3, "bbox": [200, 200, 250, 250]},
        ]
        occupancy_manager.update(tracking_3, frame_number=100)
        
        # Update with 1 person
        tracking_1 = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        occupancy_manager.update(tracking_1, frame_number=200)
        
        stats = occupancy_manager.get_current_stats()
        assert stats["occupancy"]["max_count"] == 3
    
    def test_average_occupancy(self, occupancy_manager):
        """Test average occupancy calculation."""
        # Update multiple times
        for i in range(5):
            tracking = [{"track_id": j, "bbox": [10*j, 10*j, 10*j+50, 10*j+50]} for j in range(i+1)]
            occupancy_manager.update(tracking, frame_number=i*10)
        
        stats = occupancy_manager.get_current_stats()
        # Average of [1, 2, 3, 4, 5] = 3
        assert stats["occupancy"]["average_count"] == 3.0
    
    def test_reset(self, occupancy_manager, sample_tracking_output):
        """Test resetting occupancy manager."""
        occupancy_manager.update(sample_tracking_output, frame_number=100)
        occupancy_manager.reset()
        
        assert occupancy_manager.get_current_count() == 0
        assert occupancy_manager.get_current_stats()["occupancy"]["max_count"] == 0


class TestEntryExitCounter:
    """Test cases for EntryExitCounter class."""
    
    @pytest.fixture
    def entry_exit_counter(self):
        """Create EntryExitCounter instance with explicit line config."""
        line_config = {
            "start": [0, 100],
            "end": [100, 100],
            "direction": "bidirectional",
            "debounce_frames": 5
        }
        return EntryExitCounter(
            camera_id="cam_001",
            line_config=line_config
        )
    
    def test_initialization(self, entry_exit_counter):
        """Test entry/exit counter initialization."""
        assert entry_exit_counter.camera_id == "cam_001"
        assert entry_exit_counter.get_total_entries() == 0
        assert entry_exit_counter.get_total_exits() == 0
    
    def test_entry_detection(self, entry_exit_counter):
        """Test entry detection."""
        # Track moves from below line to above line
        tracking_output = [
            {"track_id": 1, "bbox": [50, 50, 100, 100]}  # Center at (75, 75) - below line
        ]
        entry_exit_counter.update(tracking_output, frame_number=10)
        
        # Move above line
        tracking_output = [
            {"track_id": 1, "bbox": [50, 150, 100, 200]}  # Center at (75, 175) - above line
        ]
        entry_exit_counter.update(tracking_output, frame_number=20)
        
        counts = entry_exit_counter.get_counts()
        assert counts["entry_exit"]["total_entries"] == 1
        assert counts["entry_exit"]["total_exits"] == 0
    
    def test_exit_detection(self, entry_exit_counter):
        """Test exit detection."""
        # Track moves from above line to below line
        tracking_output = [
            {"track_id": 1, "bbox": [50, 150, 100, 200]}  # Center at (75, 175) - above line
        ]
        entry_exit_counter.update(tracking_output, frame_number=10)
        
        # Move below line
        tracking_output = [
            {"track_id": 1, "bbox": [50, 50, 100, 100]}  # Center at (75, 75) - below line
        ]
        entry_exit_counter.update(tracking_output, frame_number=20)
        
        counts = entry_exit_counter.get_counts()
        assert counts["entry_exit"]["total_entries"] == 0
        assert counts["entry_exit"]["total_exits"] == 1
    
    def test_no_crossing(self, entry_exit_counter):
        """Test no crossing when track stays on same side."""
        tracking_output = [
            {"track_id": 1, "bbox": [50, 50, 100, 100]}  # Below line
        ]
        entry_exit_counter.update(tracking_output, frame_number=10)
        
        # Still below line
        tracking_output = [
            {"track_id": 1, "bbox": [60, 60, 110, 110]}  # Still below line
        ]
        entry_exit_counter.update(tracking_output, frame_number=20)
        
        counts = entry_exit_counter.get_counts()
        assert counts["entry_exit"]["total_entries"] == 0
        assert counts["entry_exit"]["total_exits"] == 0
    
    def test_net_count(self, entry_exit_counter):
        """Test net count calculation."""
        # Add 2 entries
        for i in range(2):
            tracking_output = [{"track_id": i, "bbox": [50, 50, 100, 100]}]
            entry_exit_counter.update(tracking_output, frame_number=i*10)
            tracking_output = [{"track_id": i, "bbox": [50, 150, 100, 200]}]
            entry_exit_counter.update(tracking_output, frame_number=i*10+5)
        
        # Add 1 exit
        tracking_output = [{"track_id": 0, "bbox": [50, 150, 100, 200]}]
        entry_exit_counter.update(tracking_output, frame_number=30)
        tracking_output = [{"track_id": 0, "bbox": [50, 50, 100, 100]}]
        entry_exit_counter.update(tracking_output, frame_number=35)
        
        assert entry_exit_counter.get_net_count() == 1
    
    def test_reset(self, entry_exit_counter):
        """Test resetting counter."""
        tracking_output = [{"track_id": 1, "bbox": [50, 50, 100, 100]}]
        entry_exit_counter.update(tracking_output, frame_number=10)
        tracking_output = [{"track_id": 1, "bbox": [50, 150, 100, 200]}]
        entry_exit_counter.update(tracking_output, frame_number=20)
        
        entry_exit_counter.reset()
        
        assert entry_exit_counter.get_total_entries() == 0
        assert entry_exit_counter.get_total_exits() == 0


class TestZoneManager:
    """Test cases for ZoneManager class."""
    
    @pytest.fixture
    def zone_manager(self):
        """Create ZoneManager instance."""
        zones_config = {
            "main_area": {
                "id": "zone_001",
                "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)]
            }
        }
        return ZoneManager(camera_id="cam_001", zones_config=zones_config)
    
    @pytest.fixture
    def sample_tracking_output(self):
        """Create sample tracking output."""
        return [
            {"track_id": 1, "bbox": [10, 10, 30, 30]},  # Center at (20, 20) - inside
            {"track_id": 2, "bbox": [150, 150, 170, 170]},  # Center at (160, 160) - outside
        ]
    
    def test_initialization(self, zone_manager):
        """Test zone manager initialization."""
        assert zone_manager.camera_id == "cam_001"
        assert len(zone_manager._zones) == 1
    
    def test_add_zone(self, zone_manager):
        """Test adding a zone."""
        zone_manager.add_zone(
            zone_name="secondary_area",
            zone_id="zone_002",
            polygon=[(200, 200), (300, 200), (300, 300), (200, 300)]
        )
        
        assert len(zone_manager._zones) == 2
    
    def test_update_zones(self, zone_manager, sample_tracking_output):
        """Test updating zone occupancy."""
        zone_manager.update_zones(sample_tracking_output)
        
        count = zone_manager.get_zone_count("main_area")
        assert count == 1  # Only track 1 is inside
    
    def test_get_zone_track_ids(self, zone_manager, sample_tracking_output):
        """Test getting track IDs in zone."""
        zone_manager.update_zones(sample_tracking_output)
        
        track_ids = zone_manager.get_zone_track_ids("main_area")
        assert track_ids == {1}
    
    def test_get_all_zones(self, zone_manager, sample_tracking_output):
        """Test getting all zones."""
        zone_manager.update_zones(sample_tracking_output)
        
        zones = zone_manager.get_all_zones()
        assert "main_area" in zones
        assert zones["main_area"]["count"] == 1
    
    def test_get_zones_output(self, zone_manager, sample_tracking_output):
        """Test getting zones output in standardized format."""
        zone_manager.update_zones(sample_tracking_output)
        
        zones_output = zone_manager.get_zones_output()
        assert len(zones_output) == 1
        assert zones_output[0]["zone_id"] == "zone_001"
        assert zones_output[0]["count"] == 1
    
    def test_reset(self, zone_manager, sample_tracking_output):
        """Test resetting zone manager."""
        zone_manager.update_zones(sample_tracking_output)
        zone_manager.reset()
        
        assert zone_manager.get_zone_count("main_area") == 0


class TestConfigurationLoading:
    """Test cases for configuration loading."""
    
    def test_entry_exit_load_from_config(self):
        """Test EntryExitCounter loads from camera.yaml."""
        counter = EntryExitCounter(camera_id="cam_001")
        
        # Should load from camera.yaml
        assert counter.line_start == (320, 240)
        assert counter.line_end == (320, 400)
        assert counter.direction == "bidirectional"
        assert counter.debounce_frames == 10
    
    def test_zone_manager_load_from_config(self):
        """Test ZoneManager loads from camera.yaml."""
        zone_manager = ZoneManager(camera_id="cam_001")
        
        # Should load from camera.yaml
        assert len(zone_manager._zones) == 1
        assert "main_area" in zone_manager._zones
        assert zone_manager._zones["main_area"]["id"] == "zone_001"
    
    def test_entry_exit_with_custom_config(self):
        """Test EntryExitCounter with custom line config."""
        line_config = {
            "start": [100, 100],
            "end": [200, 200],
            "direction": "entry",
            "debounce_frames": 5
        }
        counter = EntryExitCounter(camera_id="cam_001", line_config=line_config)
        
        assert counter.line_start == (100, 100)
        assert counter.line_end == (200, 200)
        assert counter.direction == "entry"
        assert counter.debounce_frames == 5
    
    def test_zone_manager_with_custom_config(self):
        """Test ZoneManager with custom zones config."""
        zones_config = {
            "custom_zone": {
                "id": "zone_custom",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]
            }
        }
        zone_manager = ZoneManager(camera_id="cam_001", zones_config=zones_config)
        
        assert len(zone_manager._zones) == 1
        assert "custom_zone" in zone_manager._zones
        assert zone_manager._zones["custom_zone"]["id"] == "zone_custom"
    
    def test_entry_exit_invalid_camera_id(self):
        """Test EntryExitCounter with invalid camera ID."""
        counter = EntryExitCounter(camera_id="invalid_camera")
        
        # Should use defaults
        assert counter.line_start == (0, 0)
        assert counter.line_end == (0, 0)
    
    def test_zone_manager_invalid_camera_id(self):
        """Test ZoneManager with invalid camera ID."""
        zone_manager = ZoneManager(camera_id="invalid_camera")
        
        # Should have no zones
        assert len(zone_manager._zones) == 0


class TestMultipleCameras:
    """Test cases for multiple camera support."""
    
    def test_independent_occupancy_managers(self):
        """Test that OccupancyManager instances are independent."""
        occ1 = OccupancyManager(camera_id="cam_001")
        occ2 = OccupancyManager(camera_id="cam_002")
        
        # Update first manager
        tracking1 = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        occ1.update(tracking1, frame_number=100)
        
        # Update second manager
        tracking2 = [{"track_id": 2, "bbox": [100, 100, 150, 150]}]
        occ2.update(tracking2, frame_number=100)
        
        # Verify independence
        assert occ1.get_current_count() == 1
        assert occ2.get_current_count() == 1
        assert occ1.get_active_track_ids() == {1}
        assert occ2.get_active_track_ids() == {2}
    
    def test_independent_entry_exit_counters(self):
        """Test that EntryExitCounter instances are independent."""
        # Explicit line config matching the test's crossing coordinates.
        # Horizontal line at y=100 spanning x=0..200 covers both tracks.
        line_config = {
            "start": [0, 100],
            "end": [200, 100],
            "direction": "bidirectional",
            "debounce_frames": 5
        }
        counter1 = EntryExitCounter(camera_id="cam_001", line_config=line_config)
        counter2 = EntryExitCounter(camera_id="cam_002", line_config=line_config)
        
        # Update first counter: track 1 center moves (75,75) -> (75,175), crossing y=100
        tracking1 = [{"track_id": 1, "bbox": [50, 50, 100, 100]}]
        counter1.update(tracking1, frame_number=10)
        tracking1 = [{"track_id": 1, "bbox": [50, 150, 100, 200]}]
        counter1.update(tracking1, frame_number=20)
        
        # Update second counter: track 2 center moves (175,75) -> (175,175), crossing y=100
        tracking2 = [{"track_id": 2, "bbox": [150, 50, 200, 100]}]
        counter2.update(tracking2, frame_number=10)
        tracking2 = [{"track_id": 2, "bbox": [150, 150, 200, 200]}]
        counter2.update(tracking2, frame_number=20)
        
        # Verify independence: each counter has its own state
        assert counter1.get_total_entries() == 1
        assert counter2.get_total_entries() == 1
        assert counter1.get_total_exits() == 0
        assert counter2.get_total_exits() == 0
    
    def test_independent_zone_managers(self):
        """Test that ZoneManager instances are independent."""
        zones_config = {
            "zone_a": {"id": "zone_a", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}
        }
        
        zone1 = ZoneManager(camera_id="cam_001", zones_config=zones_config)
        zone2 = ZoneManager(camera_id="cam_002", zones_config=zones_config)
        
        # Update first zone manager
        tracking1 = [{"track_id": 1, "bbox": [10, 10, 30, 30]}]
        zone1.update_zones(tracking1)
        
        # Update second zone manager
        tracking2 = [{"track_id": 2, "bbox": [50, 50, 70, 70]}]
        zone2.update_zones(tracking2)
        
        # Verify independence
        assert zone1.get_zone_count("zone_a") == 1
        assert zone2.get_zone_count("zone_a") == 1
        assert zone1.get_zone_track_ids("zone_a") == {1}
        assert zone2.get_zone_track_ids("zone_a") == {2}
    
    def test_reset_one_camera_does_not_affect_another(self):
        """Test that resetting one camera's analytics doesn't affect another."""
        occ1 = OccupancyManager(camera_id="cam_001")
        occ2 = OccupancyManager(camera_id="cam_002")
        
        # Update both
        tracking = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        occ1.update(tracking, frame_number=100)
        occ2.update(tracking, frame_number=100)
        
        # Reset first
        occ1.reset()
        
        # Verify second is unaffected
        assert occ1.get_current_count() == 0
        assert occ2.get_current_count() == 1
        assert occ1.get_current_stats()["occupancy"]["max_count"] == 0
        assert occ2.get_current_stats()["occupancy"]["max_count"] == 1
