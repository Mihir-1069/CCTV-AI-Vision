"""
Zone Manager Module
Provides ZoneManager class for polygon-based occupancy zones.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import yaml

from .analytics_utils import AnalyticsUtils
from ..path_utils import get_config_path

logger = logging.getLogger(__name__)


class ZoneManager:
    """
    Manages polygon-based occupancy zones.
    
    Tracks people in defined zones and provides zone-specific counts.
    """
    
    def __init__(self, camera_id: str, zones_config: Optional[dict] = None, config_path: Optional[str] = None):
        """
        Initialize ZoneManager.
        
        Args:
            camera_id: Camera identifier
            zones_config: Optional dictionary of zone configurations
            config_path: Optional path to camera configuration file
        """
        self.camera_id = camera_id
        self._zones: Dict[str, Dict[str, Any]] = {}
        self._zone_track_positions: Dict[str, Dict[int, Tuple[float, float]]] = {}
        
        # Load configuration
        if config_path is None:
            config_path = get_config_path("camera.yaml")
        
        if zones_config:
            self.load_zones(zones_config)
        elif Path(config_path).exists():
            self._load_from_camera_config(config_path, camera_id)
        else:
            logger.warning(f"No configuration provided for camera {camera_id}, no zones loaded")
        
        logger.info(f"ZoneManager initialized for camera: {self.camera_id}")
    
    def _load_from_camera_config(self, config_path: str, camera_id: str):
        """
        Load zones from camera config file.
        
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
                logger.warning(f"Camera {camera_id} not found in config, no zones loaded")
                return
            
            # Load zones
            zones_config = camera_config.get('zones', {})
            self.load_zones(zones_config)
            logger.info(f"Loaded {len(self._zones)} zones from camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
    
    def load_zones(self, zones_config: dict):
        """
        Load zones from configuration.
        
        Args:
            zones_config: Dictionary of zone configurations
                Format: {zone_name: {"id": str, "polygon": [[x1,y1], [x2,y2], ...]}}
        """
        for zone_name, zone_data in zones_config.items():
            zone_id = zone_data.get('id', zone_name)
            polygon = zone_data.get('polygon', [])
            
            self._zones[zone_name] = {
                "id": zone_id,
                "name": zone_name,
                "polygon": polygon,
                "count": 0,
                "track_ids": set()
            }
            
            self._zone_track_positions[zone_name] = {}
        
        logger.info(f"Loaded {len(self._zones)} zones for camera: {self.camera_id}")
    
    def add_zone(self, zone_name: str, zone_id: str, polygon: List[Tuple[float, float]]):
        """
        Add a new zone.
        
        Args:
            zone_name: Zone name
            zone_id: Zone identifier
            polygon: List of polygon vertices [(x, y), ...]
        """
        self._zones[zone_name] = {
            "id": zone_id,
            "name": zone_name,
            "polygon": polygon,
            "count": 0,
            "track_ids": set()
        }
        self._zone_track_positions[zone_name] = {}
        
        logger.info(f"Added zone: {zone_name} (id: {zone_id})")
    
    def update_zones(self, tracking_output: List[Dict[str, Any]]):
        """
        Update zone occupancy based on tracking data.
        
        Args:
            tracking_output: List of tracking dictionaries from tracker
        """
        # Clear previous counts
        for zone_name in self._zones:
            self._zones[zone_name]["count"] = 0
            self._zones[zone_name]["track_ids"].clear()
        
        # Update each zone
        for track in tracking_output:
            track_id = track.get('track_id')
            bbox = track.get('bbox', [])
            
            if track_id < 0 or len(bbox) != 4:
                continue
            
            # Calculate bbox center
            center = AnalyticsUtils.calculate_bbox_center(bbox)
            
            # Check each zone
            for zone_name, zone_data in self._zones.items():
                polygon = zone_data["polygon"]
                
                if AnalyticsUtils.point_in_polygon(center, polygon):
                    zone_data["count"] += 1
                    zone_data["track_ids"].add(track_id)
                    self._zone_track_positions[zone_name][track_id] = center
    
    def get_zone_count(self, zone_name: str) -> int:
        """
        Get count of people in a specific zone.
        
        Args:
            zone_name: Zone name
            
        Returns:
            Number of people in zone
        """
        if zone_name not in self._zones:
            return 0
        return self._zones[zone_name]["count"]
    
    def get_zone_track_ids(self, zone_name: str) -> set:
        """
        Get track IDs in a specific zone.
        
        Args:
            zone_name: Zone name
            
        Returns:
            Set of track IDs in zone
        """
        if zone_name not in self._zones:
            return set()
        return self._zones[zone_name]["track_ids"].copy()
    
    def get_all_zones(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all zones with their current counts.
        
        Returns:
            Dictionary of zone data
        """
        zones_output = {}
        
        for zone_name, zone_data in self._zones.items():
            zones_output[zone_name] = {
                "id": zone_data["id"],
                "name": zone_data["name"],
                "count": zone_data["count"],
                "track_ids": list(zone_data["track_ids"])
            }
        
        return zones_output
    
    def get_zones_output(self) -> List[Dict[str, Any]]:
        """
        Get zones output in standardized format for JSON output.
        
        Returns:
            List of zone dictionaries
        """
        zones_list = []
        
        for zone_name, zone_data in self._zones.items():
            zones_list.append({
                "zone_id": zone_data["id"],
                "zone_name": zone_name,
                "count": zone_data["count"],
                "track_ids": list(zone_data["track_ids"])
            })
        
        return zones_list
    
    def reset(self):
        """Reset zone counts."""
        for zone_name in self._zones:
            self._zones[zone_name]["count"] = 0
            self._zones[zone_name]["track_ids"].clear()
            self._zone_track_positions[zone_name].clear()
        
        logger.info(f"ZoneManager reset for camera: {self.camera_id}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ZoneManager(camera_id={self.camera_id}, zones={len(self._zones)})"
