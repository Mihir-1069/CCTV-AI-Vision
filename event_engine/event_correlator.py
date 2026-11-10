"""
Event Correlator — Phase 11
Correlates events across frames and temporary track IDs with temporal grouping.

Design
------
- Groups related events within a temporal window (configurable seconds)
- Associates events with track IDs when available
- Handles missing or stale tracking IDs gracefully
- Avoids false identity association by using temporal proximity only
- Produces correlated event groups for downstream analysis
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from event_engine.rule_engine import RuleEvent

logger = logging.getLogger(__name__)


@dataclass
class CorrelatedEventGroup:
    """
    Group of temporally correlated events.

    Attributes
    ----------
    group_id      : Unique identifier for this correlation group
    start_time    : ISO 8601 timestamp of first event in group
    end_time      : ISO 8601 timestamp of last event in group
    camera_id     : Camera source
    track_ids     : Set of track IDs involved in this group
    events        : List of RuleEvents in this group
    severity      : Highest severity among events in group
    event_types   : Set of rule_ids involved
    """
    group_id: str
    start_time: str
    end_time: str
    camera_id: str
    track_ids: Set[int] = field(default_factory=set)
    events: List[RuleEvent] = field(default_factory=list)
    severity: str = "INFO"
    event_types: Set[str] = field(default_factory=set)

    def add_event(self, event: RuleEvent):
        """Add an event to this group and update metadata."""
        self.events.append(event)
        self.event_types.add(event.rule_id)
        
        # Update severity (CRITICAL > WARNING > INFO)
        severity_order = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
        if severity_order.get(event.severity, 0) > severity_order.get(self.severity, 0):
            self.severity = event.severity
    
    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict representation."""
        return {
            "group_id": self.group_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "camera_id": self.camera_id,
            "track_ids": list(self.track_ids),
            "event_count": len(self.events),
            "severity": self.severity,
            "event_types": list(self.event_types),
            "events": [
                {
                    "rule_id": e.rule_id,
                    "description": e.description,
                    "severity": e.severity,
                    "frame_number": e.frame_number,
                    "evidence": e.evidence
                }
                for e in self.events
            ]
        }


class EventCorrelator:
    """
    Correlates events across frames and track IDs with temporal grouping.

    Instantiate once per pipeline; call ``correlate()`` for each event.
    Groups events within a temporal window and associates with track IDs.
    """

    _DEFAULT_TEMPORAL_WINDOW = 5.0  # seconds
    _DEFAULT_MAX_GROUPS = 100

    def __init__(
        self,
        temporal_window_seconds: float = _DEFAULT_TEMPORAL_WINDOW,
        max_groups: int = _DEFAULT_MAX_GROUPS,
    ) -> None:
        """
        Initialise EventCorrelator.

        Args:
            temporal_window_seconds: Group events within this time window.
            max_groups              : Maximum number of active groups to retain.
        """
        self._temporal_window = temporal_window_seconds
        self._max_groups = max_groups
        self._active_groups: List[CorrelatedEventGroup] = []
        self._group_counter = 0

        logger.info(
            "EventCorrelator initialised (temporal_window=%.1fs, max_groups=%d)",
            temporal_window_seconds,
            max_groups,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlate(
        self,
        event: RuleEvent,
        track_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[CorrelatedEventGroup]:
        """
        Correlate an event with existing groups or create a new group.

        Args:
            event     : RuleEvent from RuleEngine.evaluate()
            track_id  : Optional track ID associated with this event
            timestamp : Optional ISO 8601 timestamp (defaults to current time)

        Returns:
            CorrelatedEventGroup containing this event, or None if no group created.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        event_time = datetime.fromisoformat(timestamp)

        # Try to find an existing group to add this event to
        matched_group = self._find_matching_group(event, track_id, event_time)

        if matched_group:
            # Add to existing group
            matched_group.add_event(event)
            if track_id is not None:
                matched_group.track_ids.add(track_id)
            
            # Update end time
            matched_group.end_time = timestamp
            
            logger.debug(
                "Added event to existing group %s (now %d events)",
                matched_group.group_id,
                len(matched_group.events),
            )
            return matched_group
        else:
            # Create new group
            new_group = self._create_group(event, track_id, timestamp)
            self._active_groups.append(new_group)
            self._cleanup_old_groups(event_time)
            
            logger.debug(
                "Created new correlation group %s for event %s",
                new_group.group_id,
                event.rule_id,
            )
            return new_group

    def get_active_groups(self) -> List[CorrelatedEventGroup]:
        """
        Get all currently active correlation groups.

        Returns:
            List of CorrelatedEventGroup objects.
        """
        return self._active_groups.copy()

    def get_groups_by_camera(self, camera_id: str) -> List[CorrelatedEventGroup]:
        """
        Get correlation groups for a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of CorrelatedEventGroup objects for the camera.
        """
        return [g for g in self._active_groups if g.camera_id == camera_id]

    def get_groups_by_track(self, track_id: int) -> List[CorrelatedEventGroup]:
        """
        Get correlation groups involving a specific track ID.

        Args:
            track_id: Track identifier

        Returns:
            List of CorrelatedEventGroup objects involving the track.
        """
        return [g for g in self._active_groups if track_id in g.track_ids]

    def reset(self):
        """Clear all correlation groups."""
        self._active_groups.clear()
        self._group_counter = 0
        logger.info("EventCorrelator reset")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _find_matching_group(
        self,
        event: RuleEvent,
        track_id: Optional[int],
        event_time: datetime,
    ) -> Optional[CorrelatedEventGroup]:
        """
        Find an existing group that matches this event.

        Matching criteria:
        1. Same camera ID
        2. Within temporal window
        3. Same track ID (if available) OR same rule type

        Args:
            event     : RuleEvent to match
            track_id  : Optional track ID
            event_time : Event timestamp as datetime

        Returns:
            Matching CorrelatedEventGroup or None.
        """
        for group in self._active_groups:
            # Check camera match
            if group.camera_id != event.camera_id:
                continue

            # Check temporal window
            group_end_time = datetime.fromisoformat(group.end_time)
            time_diff = abs((event_time - group_end_time).total_seconds())
            if time_diff > self._temporal_window:
                continue

            # Check track ID match (if available)
            if track_id is not None and track_id in group.track_ids:
                return group

            # Check rule type match (same rule within temporal window)
            if event.rule_id in group.event_types:
                return group

        return None

    def _create_group(
        self,
        event: RuleEvent,
        track_id: Optional[int],
        timestamp: str,
    ) -> CorrelatedEventGroup:
        """
        Create a new correlation group.

        Args:
            event     : RuleEvent to add
            track_id  : Optional track ID
            timestamp : Event timestamp

        Returns:
            New CorrelatedEventGroup.
        """
        self._group_counter += 1
        group_id = f"group_{self._group_counter}"

        group = CorrelatedEventGroup(
            group_id=group_id,
            start_time=timestamp,
            end_time=timestamp,
            camera_id=event.camera_id,
        )

        group.add_event(event)
        if track_id is not None:
            group.track_ids.add(track_id)

        return group

    def _cleanup_old_groups(self, current_time: datetime):
        """
        Remove groups that are too old or exceed max limit.

        Args:
            current_time: Current datetime for age comparison.
        """
        # Remove groups older than 2x temporal window
        cutoff_time = current_time.timestamp() - (2 * self._temporal_window)
        
        self._active_groups = [
            g for g in self._active_groups
            if datetime.fromisoformat(g.end_time).timestamp() > cutoff_time
        ]

        # Enforce max groups limit (remove oldest)
        if len(self._active_groups) > self._max_groups:
            self._active_groups = self._active_groups[-self._max_groups:]
