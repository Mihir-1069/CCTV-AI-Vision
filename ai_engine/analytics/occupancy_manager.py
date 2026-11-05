"""
Occupancy Manager Module
Provides OccupancyManager class for tracking people count in camera view.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .analytics_utils import AnalyticsUtils

logger = logging.getLogger(__name__)


class OccupancyManager:
    """
    Manages occupancy analytics for a camera view.
    
    Tracks current number of people, occupancy history,
    and provides statistics over time.
    """
    
    def __init__(self, camera_id: str, config: Optional[dict] = None):
        """
        Initialize OccupancyManager.
        
        Args:
            camera_id: Camera identifier
            config: Optional configuration dictionary
        """
        self.camera_id = camera_id
        
        # Current state
        self._current_count: int = 0
        self._active_track_ids: set = set()
        
        # History
        self._occupancy_history: List[Dict[str, Any]] = []
        self._max_occupancy: int = 0
        self._min_occupancy: int = 0
        self._total_occupancy_sum: int = 0
        self._occupancy_sample_count: int = 0
        
        # Configuration
        self.max_capacity: Optional[int] = config.get('capacity', None) if config else None
        self.alert_threshold: Optional[int] = config.get('alert_threshold', None) if config else None
        
        logger.info(f"OccupancyManager initialized for camera: {camera_id}")
    
    def update(
        self,
        tracking_output: List[Dict[str, Any]],
        frame_number: int,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update occupancy with new tracking data.
        
        Args:
            tracking_output: List of tracking dictionaries from tracker
            frame_number: Current frame number
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Occupancy statistics dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract active track IDs
        current_track_ids = set()
        for track in tracking_output:
            track_id = track.get('track_id')
            if track_id >= 0:  # Valid track ID
                current_track_ids.add(track_id)
        
        # Update current count
        self._current_count = len(current_track_ids)
        self._active_track_ids = current_track_ids
        
        # Update statistics
        if self._occupancy_sample_count == 0:
            self._min_occupancy = self._current_count
        
        self._max_occupancy = max(self._max_occupancy, self._current_count)
        self._min_occupancy = min(self._min_occupancy, self._current_count)
        self._total_occupancy_sum += self._current_count
        self._occupancy_sample_count += 1
        
        # Add to history
        history_entry = {
            "timestamp": AnalyticsUtils.format_timestamp(timestamp),
            "frame_number": frame_number,
            "count": self._current_count,
            "track_ids": list(current_track_ids)
        }
        self._occupancy_history.append(history_entry)
        
        # Limit history size (keep last 1000 entries)
        if len(self._occupancy_history) > 1000:
            self._occupancy_history = self._occupancy_history[-1000:]
        
        return self.get_current_stats(timestamp, frame_number)
    
    def get_current_count(self) -> int:
        """
        Get current occupancy count.
        
        Returns:
            Current number of people
        """
        return self._current_count
    
    def get_active_track_ids(self) -> set:
        """
        Get active track IDs.
        
        Returns:
            Set of active track IDs
        """
        return self._active_track_ids.copy()
    
    def get_current_stats(
        self,
        timestamp: Optional[datetime] = None,
        frame_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get current occupancy statistics.
        
        Args:
            timestamp: Optional timestamp
            frame_number: Optional frame number
            
        Returns:
            Dictionary with occupancy statistics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        avg_count = (
            self._total_occupancy_sum / self._occupancy_sample_count
            if self._occupancy_sample_count > 0
            else 0
        )
        
        stats = {
            "camera_id": self.camera_id,
            "timestamp": AnalyticsUtils.format_timestamp(timestamp),
            "frame_number": frame_number if frame_number is not None else 0,
            "occupancy": {
                "current_count": self._current_count,
                "max_count": self._max_occupancy,
                "min_count": self._min_occupancy,
                "average_count": avg_count
            }
        }
        
        # Add capacity info if configured
        if self.max_capacity is not None:
            stats["occupancy"]["capacity"] = self.max_capacity
            stats["occupancy"]["capacity_percentage"] = (
                (self._current_count / self.max_capacity) * 100
                if self.max_capacity > 0
                else 0
            )
        
        # Add alert threshold if configured
        if self.alert_threshold is not None:
            stats["occupancy"]["alert_threshold"] = self.alert_threshold
            stats["occupancy"]["alert_triggered"] = (
                self._current_count >= self.alert_threshold
            )
        
        return stats
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get occupancy history.
        
        Args:
            limit: Maximum number of entries to return (None = all)
            
        Returns:
            List of history entries
        """
        if limit is None:
            return self._occupancy_history.copy()
        return self._occupancy_history[-limit:]
    
    def reset(self):
        """Reset occupancy statistics."""
        self._current_count = 0
        self._active_track_ids.clear()
        self._occupancy_history.clear()
        self._max_occupancy = 0
        self._min_occupancy = 0
        self._total_occupancy_sum = 0
        self._occupancy_sample_count = 0
        
        logger.info(f"OccupancyManager reset for camera: {self.camera_id}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OccupancyManager(camera_id={self.camera_id}, current_count={self._current_count})"
