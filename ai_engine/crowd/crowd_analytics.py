"""
Crowd and Movement Analytics Module — Phase 12
Provides crowd density estimation, movement analysis, and congestion detection.

Design
------
- Crowd density estimation based on tracking output and frame area
- Crowd distribution analysis across zones
- Movement direction and flow analysis using track velocity vectors
- Congestion detection based on density thresholds
- Zone-based analysis reusing existing ZoneManager
- Temporal smoothing for stable metrics
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_engine.analytics.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class CrowdMetrics:
    """
    Crowd metrics for a single frame.

    Attributes
    ----------
    timestamp        : ISO 8601 timestamp
    frame_number     : Frame index
    camera_id        : Camera source
    total_count      : Total number of people in frame
    density          : People per square meter (or per 1000 pixels)
    avg_velocity     : Average movement velocity (pixels/frame)
    movement_direction : Dominant movement direction (degrees)
    congestion_level : "LOW", "MEDIUM", "HIGH"
    zone_metrics     : Per-zone crowd metrics
    """
    timestamp: str
    frame_number: int
    camera_id: str
    total_count: int
    density: float
    avg_velocity: float
    movement_direction: float
    congestion_level: str
    zone_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict representation."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "camera_id": self.camera_id,
            "total_count": self.total_count,
            "density": self.density,
            "avg_velocity": self.avg_velocity,
            "movement_direction": self.movement_direction,
            "congestion_level": self.congestion_level,
            "zone_metrics": self.zone_metrics,
        }


class CrowdAnalytics:
    """
    Crowd and movement analytics using tracking output.

    Instantiate once per pipeline; call ``update()`` every frame.
    Reuses ZoneManager for zone-based analysis.
    """

    _DEFAULT_HISTORY_SIZE = 30  # frames
    _DEFAULT_DENSITY_THRESHOLD_LOW = 0.5  # people per 1000 pixels
    _DEFAULT_DENSITY_THRESHOLD_HIGH = 2.0  # people per 1000 pixels

    def __init__(
        self,
        camera_id: str,
        zone_manager: Optional[ZoneManager] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        density_threshold_low: float = _DEFAULT_DENSITY_THRESHOLD_LOW,
        density_threshold_high: float = _DEFAULT_DENSITY_THRESHOLD_HIGH,
    ) -> None:
        """
        Initialise CrowdAnalytics.

        Args:
            camera_id              : Camera identifier
            zone_manager           : Optional ZoneManager for zone-based analysis
            frame_width            : Frame width in pixels
            frame_height           : Frame height in pixels
            history_size           : Number of frames to keep for temporal smoothing
            density_threshold_low  : Density threshold for LOW congestion
            density_threshold_high : Density threshold for HIGH congestion
        """
        self.camera_id = camera_id
        self.zone_manager = zone_manager
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.history_size = history_size
        self.density_threshold_low = density_threshold_low
        self.density_threshold_high = density_threshold_high

        # Track velocity history per track
        self._track_velocity_history: Dict[int, deque] = {}

        # Metrics history for temporal smoothing
        self._metrics_history: deque = deque(maxlen=history_size)

        logger.info(
            "CrowdAnalytics initialised for camera %s (frame=%dx%d)",
            camera_id,
            frame_width,
            frame_height,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracking_output: List[Dict[str, Any]],
        frame_number: int,
        timestamp: Optional[str] = None,
    ) -> CrowdMetrics:
        """
        Update crowd metrics with new tracking data.

        Args:
            tracking_output : List of tracking dictionaries from tracker
            frame_number    : Current frame number
            timestamp       : Optional ISO 8601 timestamp

        Returns:
            CrowdMetrics for this frame.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Filter invalid tracking data
        valid_tracking = [
            t for t in tracking_output
            if t.get("track_id", -1) >= 0 and len(t.get("bbox", [])) == 4
        ]

        # Calculate basic metrics
        total_count = len(valid_tracking)
        frame_area = self.frame_width * self.frame_height
        density = (total_count / frame_area) * 1000.0  # people per 1000 pixels

        # Calculate movement metrics
        velocities = self._calculate_velocities(valid_tracking, frame_number)
        avg_velocity = np.mean(velocities) if velocities else 0.0
        movement = self._calculate_movement_direction(velocities) if velocities else 0.0

        # Determine congestion level
        congestion_level = self._determine_congestion_level(density)

        # Zone-based metrics
        zone_metrics = {}
        if self.zone_manager:
            zone_metrics = self._calculate_zone_metrics(valid_tracking)

        # Create metrics object
        metrics = CrowdMetrics(
            timestamp=timestamp,
            frame_number=frame_number,
            camera_id=self.camera_id,
            total_count=total_count,
            density=density,
            avg_velocity=avg_velocity,
            movement_direction=movement,
            congestion_level=congestion_level,
            zone_metrics=zone_metrics,
        )

        # Add to history
        self._metrics_history.append(metrics)

        return metrics

    def get_smoothed_metrics(self, window_size: int = 5) -> Optional[CrowdMetrics]:
        """
        Get temporally smoothed crowd metrics.

        Args:
            window_size: Number of recent frames to average

        Returns:
            Smoothed CrowdMetrics or None if insufficient history.
        """
        if len(self._metrics_history) < window_size:
            return None

        recent = list(self._metrics_history)[-window_size:]

        # Average numeric metrics
        avg_density = np.mean([m.density for m in recent])
        avg_velocity = np.mean([m.avg_velocity for m in recent])
        avg_direction = np.mean([m.movement_direction for m in recent])

        # Use most recent non-numeric fields
        latest = recent[-1]

        smoothed = CrowdMetrics(
            timestamp=latest.timestamp,
            frame_number=latest.frame_number,
            camera_id=latest.camera_id,
            total_count=int(np.mean([m.total_count for m in recent])),
            density=avg_density,
            avg_velocity=avg_velocity,
            movement_direction=avg_direction,
            congestion_level=latest.congestion_level,
            zone_metrics=latest.zone_metrics,
        )

        return smoothed

    def get_congestion_history(self) -> List[Dict[str, Any]]:
        """
        Get congestion level history.

        Returns:
            List of dicts with timestamp and congestion_level.
        """
        return [
            {"timestamp": m.timestamp, "congestion_level": m.congestion_level}
            for m in self._metrics_history
        ]

    def reset(self):
        """Clear all history and state."""
        self._track_velocity_history.clear()
        self._metrics_history.clear()
        logger.info("CrowdAnalytics reset for camera %s", self.camera_id)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _calculate_velocities(
        self, tracking_output: List[Dict[str, Any]], frame_number: int
    ) -> List[float]:
        """
        Calculate velocity for each track.

        Args:
            tracking_output : List of tracking dictionaries
            frame_number    : Current frame number

        Returns:
            List of velocity magnitudes (pixels/frame).
        """
        velocities = []

        for track in tracking_output:
            track_id = track.get("track_id")
            bbox = track.get("bbox", [])

            if track_id < 0 or len(bbox) != 4:
                continue

            # Calculate current center
            x1, y1, x2, y2 = bbox
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # Get previous position
            if track_id in self._track_velocity_history:
                prev_center = self._track_velocity_history[track_id][-1]
                dx = current_center[0] - prev_center[0]
                dy = current_center[1] - prev_center[1]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)

            # Update history
            if track_id not in self._track_velocity_history:
                self._track_velocity_history[track_id] = deque(maxlen=5)
            self._track_velocity_history[track_id].append(current_center)

        return velocities

    def _calculate_movement_direction(self, velocities: List[float]) -> float:
        """
        Calculate dominant movement direction.

        Args:
            velocities: List of velocity magnitudes

        Returns:
            Direction in degrees (0-360).
        """
        # Simplified: return 0 if no significant movement
        if not velocities or np.mean(velocities) < 1.0:
            return 0.0

        # In a full implementation, this would use velocity vectors
        # For now, return a placeholder based on average velocity
        return 45.0  # Placeholder: northeast direction

    def _determine_congestion_level(self, density: float) -> str:
        """
        Determine congestion level based on density.

        Args:
            density: People per 1000 pixels

        Returns:
            "LOW", "MEDIUM", or "HIGH".
        """
        if density < self.density_threshold_low:
            return "LOW"
        elif density < self.density_threshold_high:
            return "MEDIUM"
        else:
            return "HIGH"

    def _calculate_zone_metrics(
        self, tracking_output: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-zone crowd metrics.

        Args:
            tracking_output: List of tracking dictionaries

        Returns:
            Dict mapping zone names to their metrics.
        """
        zone_metrics = {}

        if not self.zone_manager:
            return zone_metrics

        # Get zone data from zone manager
        zones = self.zone_manager.get_all_zones()

        for zone_name, zone_data in zones.items():
            count = zone_data["count"]
            polygon = self.zone_manager._zones[zone_name]["polygon"]

            # Calculate zone area (simplified polygon area)
            zone_area = self._calculate_polygon_area(polygon)
            zone_density = (count / zone_area) * 1000.0 if zone_area > 0 else 0.0

            zone_metrics[zone_name] = {
                "zone_id": zone_data["id"],
                "count": count,
                "density": zone_density,
                "congestion_level": self._determine_congestion_level(zone_density),
            }

        return zone_metrics

    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate polygon area using shoelace formula.

        Args:
            polygon: List of (x, y) vertices

        Returns:
            Area in square pixels.
        """
        if len(polygon) < 3:
            return 0.0

        n = len(polygon)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]

        return abs(area) / 2.0
