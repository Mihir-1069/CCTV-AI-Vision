"""
Unit tests for Event Correlator module.
"""

import pytest
from datetime import datetime, timezone, timedelta

from event_engine.event_correlator import EventCorrelator, CorrelatedEventGroup
from event_engine.rule_engine import RuleEvent


class TestCorrelatedEventGroup:
    """Test cases for CorrelatedEventGroup class."""

    @pytest.fixture
    def sample_event(self):
        """Create a sample RuleEvent."""
        return RuleEvent(
            rule_id="test_rule",
            description="Test rule triggered",
            severity="WARNING",
            frame_number=100,
            camera_id="cam_001",
            evidence={"value": 10}
        )
    
    @pytest.fixture
    def event_group(self, sample_event):
        """Create a CorrelatedEventGroup with a sample event."""
        group = CorrelatedEventGroup(
            group_id="group_1",
            start_time="2026-11-05T10:00:00Z",
            end_time="2026-11-05T10:00:05Z",
            camera_id="cam_001"
        )
        group.add_event(sample_event)
        return group
    
    def test_add_event(self, event_group, sample_event):
        """Test adding an event to a group."""
        initial_count = len(event_group.events)
        event_group.add_event(sample_event)
        assert len(event_group.events) == initial_count + 1
    
    def test_severity_update(self, event_group):
        """Test severity updates when higher severity event is added."""
        assert event_group.severity == "WARNING"
        
        critical_event = RuleEvent(
            rule_id="critical_rule",
            description="Critical event",
            severity="CRITICAL",
            frame_number=101,
            camera_id="cam_001"
        )
        event_group.add_event(critical_event)
        assert event_group.severity == "CRITICAL"
    
    def test_to_dict(self, event_group):
        """Test JSON serialization."""
        result = event_group.to_dict()
        assert isinstance(result, dict)
        assert result["group_id"] == "group_1"
        assert result["camera_id"] == "cam_001"
        assert result["event_count"] == 1
        assert isinstance(result["track_ids"], list)
        assert isinstance(result["event_types"], list)


class TestEventCorrelator:
    """Test cases for EventCorrelator class."""
    
    @pytest.fixture
    def correlator(self):
        """Create an EventCorrelator instance."""
        return EventCorrelator(temporal_window_seconds=5.0, max_groups=10)
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample RuleEvent."""
        return RuleEvent(
            rule_id="test_rule",
            description="Test rule triggered",
            severity="WARNING",
            frame_number=100,
            camera_id="cam_001",
            evidence={"value": 10}
        )
    
    def test_initialization(self, correlator):
        """Test EventCorrelator initialization."""
        assert correlator._temporal_window == 5.0
        assert correlator._max_groups == 10
        assert len(correlator._active_groups) == 0
    
    def test_correlate_creates_new_group(self, correlator, sample_event):
        """Test correlating an event creates a new group."""
        group = correlator.correlate(sample_event, track_id=1)
        assert group is not None
        assert group.group_id == "group_1"
        assert len(group.events) == 1
        assert 1 in group.track_ids
    
    def test_correlate_adds_to_existing_group(self, correlator, sample_event):
        """Test correlating similar event adds to existing group."""
        # First event
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        # Second similar event within temporal window
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="Test rule triggered again",
            severity="WARNING",
            frame_number=101,
            camera_id="cam_001"
        )
        group = correlator.correlate(sample_event2, track_id=1, timestamp="2026-11-05T10:00:02Z")
        
        assert group is not None
        assert len(group.events) == 2
    
    def test_correlate_different_camera_creates_new_group(self, correlator, sample_event):
        """Test events from different cameras create separate groups."""
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="Test rule on different camera",
            severity="WARNING",
            frame_number=101,
            camera_id="cam_002"
        )
        group2 = correlator.correlate(sample_event2, track_id=2, timestamp="2026-11-05T10:00:02Z")
        
        assert group2 is not None
        assert group2.camera_id == "cam_002"
        assert len(correlator._active_groups) == 2
    
    def test_correlate_outside_temporal_window(self, correlator, sample_event):
        """Test events outside temporal window create new groups."""
        correlator._temporal_window = 2.0  # Set shorter window for test
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="Test rule much later",
            severity="WARNING",
            frame_number=200,
            camera_id="cam_001"
        )
        # 5 seconds later, outside 2-second window
        group2 = correlator.correlate(sample_event2, track_id=1, timestamp="2026-11-05T10:00:05Z")
        
        assert group2 is not None
        # Old group should be cleaned up, so only 1 group remains
        assert len(correlator._active_groups) == 1
    
    def test_get_active_groups(self, correlator, sample_event):
        """Test getting active groups."""
        correlator.correlate(sample_event, track_id=1)
        groups = correlator.get_active_groups()
        assert len(groups) == 1
        assert groups[0].group_id == "group_1"
    
    def test_get_groups_by_camera(self, correlator, sample_event):
        """Test filtering groups by camera."""
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="Test on cam_002",
            severity="WARNING",
            frame_number=101,
            camera_id="cam_002"
        )
        correlator.correlate(sample_event2, track_id=2, timestamp="2026-11-05T10:00:02Z")
        
        cam001_groups = correlator.get_groups_by_camera("cam_001")
        cam002_groups = correlator.get_groups_by_camera("cam_002")
        
        assert len(cam001_groups) == 1
        assert len(cam002_groups) == 1
    
    def test_get_groups_by_track(self, correlator, sample_event):
        """Test filtering groups by track ID."""
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="Test for track 2",
            severity="WARNING",
            frame_number=101,
            camera_id="cam_001"
        )
        correlator.correlate(sample_event2, track_id=2, timestamp="2026-11-05T10:00:02Z")
        
        track1_groups = correlator.get_groups_by_track(1)
        track2_groups = correlator.get_groups_by_track(2)
        
        assert len(track1_groups) == 1
        assert len(track2_groups) == 1
    
    def test_reset(self, correlator, sample_event):
        """Test resetting the correlator."""
        correlator.correlate(sample_event, track_id=1)
        assert len(correlator._active_groups) == 1
        
        correlator.reset()
        assert len(correlator._active_groups) == 0
        assert correlator._group_counter == 0
    
    def test_cleanup_old_groups(self, correlator, sample_event):
        """Test automatic cleanup of old groups."""
        correlator._temporal_window = 2.0
        
        # Old event
        correlator.correlate(sample_event, track_id=1, timestamp="2026-11-05T10:00:00Z")
        
        # New event after cleanup threshold
        sample_event2 = RuleEvent(
            rule_id="test_rule",
            description="New event",
            severity="WARNING",
            frame_number=200,
            camera_id="cam_001"
        )
        correlator.correlate(sample_event2, track_id=2, timestamp="2026-11-05T10:00:10Z")
        
        # Old group should be cleaned up
        assert len(correlator._active_groups) == 1
