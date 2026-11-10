"""
Entry/Exit Counter Module
Provides EntryExitCounter class for counting people crossing virtual lines.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import yaml

from .analytics_utils import AnalyticsUtils
from ..path_utils import get_config_path

logger = logging.getLogger(__name__)


class EntryExitCounter:
    """
    Counts people crossing virtual lines.
    
    Detects entry and exit events based on line crossings
    with configurable debounce logic.
    """
    
    def __init__(
        self,
        camera_id: str,
        line_config: Optional[dict] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize EntryExitCounter.
        
        Args:
            camera_id: Camera identifier
            line_config: Optional line configuration dictionary
            config_path: Optional path to camera configuration file
        """
        self.camera_id = camera_id
        self.line_start: Tuple[float, float] = (0, 0)
        self.line_end: Tuple[float, float] = (0, 0)
        self.direction: str = "bidirectional"
        self.debounce_frames: int = 10
        
        # Load configuration
        if config_path is None:
            config_path = get_config_path("camera.yaml")
        
        if line_config:
            self._apply_line_config(line_config)
        elif Path(config_path).exists():
            self._load_from_camera_config(config_path, camera_id)
        else:
            logger.warning(f"No configuration provided for camera {camera_id}, using defaults")
        
        # Counters
        self._total_entries: int = 0
        self._total_exits: int = 0
        
        # Track previous positions for crossing detection
        self._track_positions: Dict[int, Tuple[float, float]] = {}
        
        # Track cooldown for debounce
        self._track_cooldown: Dict[int, int] = {}
        
        logger.info(f"EntryExitCounter initialized for camera: {camera_id}")
        logger.info(f"Line: {self.line_start} -> {self.line_end}, Direction: {self.direction}")
    
    def _load_from_camera_config(self, config_path: str, camera_id: str):
        """
        Load line configuration from camera config file.
        
        Args:
            config_path: Path to camera configuration file
            camera_id: Camera identifier
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            cameras = config.get('cameras', {})
            
            # Find camera configuration
            camera_config = None
            for cam_key, cam_data in cameras.items():
                if cam_data.get('id') == camera_id or cam_key == camera_id:
                    camera_config = cam_data
                    break
            
            if camera_config is None:
                logger.warning(f"Camera {camera_id} not found in config, using defaults")
                return
            
            # Load counting lines
            counting_lines = camera_config.get('counting_lines', {})
            
            if counting_lines:
                # Use first counting line
                line_name = list(counting_lines.keys())[0]
                line_data = counting_lines[line_name]
                
                self._apply_line_config(line_data)
                logger.info(f"Loaded line config from camera {camera_id}: {line_name}")
            
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
    
    def _apply_line_config(self, line_config: dict):
        """
        Apply line configuration.
        
        Args:
            line_config: Line configuration dictionary
        """
        if 'start' in line_config:
            self.line_start = tuple(line_config['start'])
        if 'end' in line_config:
            self.line_end = tuple(line_config['end'])
        if 'direction' in line_config:
            self.direction = line_config['direction']
        if 'debounce_frames' in line_config:
            self.debounce_frames = line_config['debounce_frames']
    
    def update(self, tracking_output: List[Dict[str, Any]], frame_number: int):
        """
        Update entry/exit counts with new tracking data.
        
        Args:
            tracking_output: List of tracking dictionaries from tracker
            frame_number: Current frame number
        """
        for track in tracking_output:
            track_id = track.get('track_id')
            bbox = track.get('bbox', [])
            
            if track_id < 0 or len(bbox) != 4:
                continue
            
            # Calculate current position (bbox center)
            current_pos = AnalyticsUtils.calculate_bbox_center(bbox)
            
            # Check if track has previous position
            if track_id in self._track_positions:
                prev_pos = self._track_positions[track_id]
                
                # Check cooldown
                if track_id in self._track_cooldown:
                    if frame_number - self._track_cooldown[track_id] < self.debounce_frames:
                        # Still in cooldown, skip
                        self._track_positions[track_id] = current_pos
                        continue
                
                # Detect line crossing
                crossing = AnalyticsUtils.detect_line_crossing(
                    prev_pos,
                    current_pos,
                    self.line_start,
                    self.line_end
                )
                
                # Update counters based on crossing and direction
                if crossing == "entry":
                    if self.direction in ["entry", "bidirectional"]:
                        self._total_entries += 1
                        self._track_cooldown[track_id] = frame_number
                        logger.debug(f"Entry detected: Track {track_id} at frame {frame_number}")
                
                elif crossing == "exit":
                    if self.direction in ["exit", "bidirectional"]:
                        self._total_exits += 1
                        self._track_cooldown[track_id] = frame_number
                        logger.debug(f"Exit detected: Track {track_id} at frame {frame_number}")
            
            # Update position
            self._track_positions[track_id] = current_pos
            
            # Clean up old positions (tracks that haven't been seen recently)
            # This is handled by the tracker's track buffer, but we can also clean up here
            # if needed based on frame_number
    
    def get_counts(self) -> Dict[str, Any]:
        """
        Get current entry/exit counts.
        
        Returns:
            Dictionary with entry/exit statistics
        """
        net_count = self._total_entries - self._total_exits
        
        return {
            "camera_id": self.camera_id,
            "entry_exit": {
                "total_entries": self._total_entries,
                "total_exits": self._total_exits,
                "net_count": net_count
            }
        }
    
    def get_total_entries(self) -> int:
        """Get total entry count."""
        return self._total_entries
    
    def get_total_exits(self) -> int:
        """Get total exit count."""
        return self._total_exits
    
    def get_net_count(self) -> int:
        """Get net count (entries - exits)."""
        return self._total_entries - self._total_exits
    
    def reset(self):
        """Reset entry/exit counters."""
        self._total_entries = 0
        self._total_exits = 0
        self._track_positions.clear()
        self._track_cooldown.clear()
        
        logger.info(f"EntryExitCounter reset for camera: {self.camera_id}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EntryExitCounter(camera_id={self.camera_id}, entries={self._total_entries}, exits={self._total_exits})"
