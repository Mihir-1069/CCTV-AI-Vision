"""
ByteTrack Multi-Object Tracking Module
Provides ByteTrackManager class for anonymous person tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
import yaml

try:
    from supervision.tracker.byte_tracker import BYTETracker
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Supervision BYTETracker not available. Using simple IoU tracker instead.")

from .tracking_utils import TrackingUtils
from ..path_utils import get_config_path

logger = logging.getLogger(__name__)


class SimpleIOUTracker:
    """
    Simple IoU-based tracker as fallback when ByteTrack is not available.
    """
    
    def __init__(self, iou_threshold: float = 0.5, max_age: int = 30):
        """
        Initialize simple IoU tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_age: Maximum frames to keep track alive without detection
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}  # track_id -> {bbox, age, last_seen}
    
    def update(self, detections: np.ndarray, image_size: list) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: numpy array of shape (N, 6) with [x1, y1, x2, y2, confidence, class_id]
            image_size: [width, height]
            
        Returns:
            numpy array with track_id in column 4
        """
        if len(detections) == 0:
            # Age all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return detections
        
        # Extract bboxes
        new_boxes = detections[:, :4]
        
        # Match new detections to existing tracks
        matched_indices = self._match_detections(new_boxes)
        
        # Update matched tracks
        for new_idx, track_id in matched_indices:
            self.tracks[track_id]['bbox'] = new_boxes[new_idx]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['last_seen'] = 0
        
        # Create new tracks for unmatched detections
        used_new_indices = {idx for idx, _ in matched_indices}
        for new_idx in range(len(new_boxes)):
            if new_idx not in used_new_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': new_boxes[new_idx],
                    'age': 0,
                    'last_seen': 0
                }
                matched_indices.append((new_idx, track_id))
        
        # Age unmatched tracks
        all_track_ids = {tid for _, tid in matched_indices}
        for track_id in list(self.tracks.keys()):
            if track_id not in all_track_ids:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Add track IDs to detections
        result = detections.copy()
        track_ids = np.zeros(len(detections), dtype=np.int32)
        
        for new_idx, track_id in matched_indices:
            track_ids[new_idx] = track_id
        
        result[:, 4] = track_ids
        
        return result
    
    def _match_detections(self, new_boxes: np.ndarray) -> list:
        """
        Match new detections to existing tracks using IoU.
        
        Args:
            new_boxes: New bounding boxes
            
        Returns:
            List of (new_idx, track_id) tuples
        """
        matched = []
        used_track_ids = set()
        
        for new_idx, new_box in enumerate(new_boxes):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                
                iou = TrackingUtils.calculate_iou(
                    new_box.tolist(),
                    track['bbox'].tolist()
                )
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched.append((new_idx, best_track_id))
                used_track_ids.add(best_track_id)
        
        return matched


class ByteTrackManager:
    """
    Multi-object tracker using ByteTrack algorithm.
    
    Associates detections across consecutive frames to maintain
    temporary tracking IDs for anonymous person tracking.
    """
    
    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        """
        Initialize ByteTrackManager.
        
        Args:
            config: Configuration dictionary (overrides config file)
            config_path: Path to YAML configuration file
        """
        self.tracker: Optional[BYTETracker] = None
        self.tracker_type: str = "bytetrack"
        self.track_thresh: float = 0.5
        self.track_buffer: int = 30
        self.match_thresh: float = 0.8
        self.fuse_score: bool = True
        self.min_box_area: int = 10
        self.frame_rate: int = 30
        
        # Track management
        self._track_id_counter: int = 0
        self._track_history: Dict[int, List[Dict[str, Any]]] = {}
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._frame_number: int = 0
        
        # Performance tracking
        self._tracking_times: List[float] = []
        self._total_tracks_created: int = 0
        
        # Load configuration
        if config_path is None:
            config_path = get_config_path("model.yaml")
        
        if Path(config_path).exists():
            self._load_config(config_path, config)
        elif config:
            self._apply_config(config)
        else:
            logger.warning("No configuration provided, using defaults")
        
        # Initialize tracker
        self.initialize_tracker()
    
    def _load_config(self, config_path: str, override_config: Optional[dict] = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            override_config: Optional override configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply tracking configuration
            tracking_config = config.get('tracking', {})
            self.tracker_type = tracking_config.get('tracker_type', self.tracker_type)
            self.track_thresh = tracking_config.get('track_thresh', self.track_thresh)
            self.track_buffer = tracking_config.get('track_buffer', self.track_buffer)
            self.match_thresh = tracking_config.get('match_thresh', self.match_thresh)
            self.fuse_score = tracking_config.get('fuse_score', self.fuse_score)
            self.min_box_area = tracking_config.get('min_box_area', self.min_box_area)
            self.frame_rate = tracking_config.get('frame_rate', self.frame_rate)
            
            # Apply overrides
            if override_config:
                self._apply_config(override_config)
            
            logger.info(f"Tracking configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _apply_config(self, config: dict):
        """
        Apply configuration dictionary.
        
        Args:
            config: Configuration dictionary
        """
        if 'tracker_type' in config:
            self.tracker_type = config['tracker_type']
        if 'track_thresh' in config:
            self.track_thresh = config['track_thresh']
        if 'track_buffer' in config:
            self.track_buffer = config['track_buffer']
        if 'match_thresh' in config:
            self.match_thresh = config['match_thresh']
        if 'fuse_score' in config:
            self.fuse_score = config['fuse_score']
        if 'min_box_area' in config:
            self.min_box_area = config['min_box_area']
        if 'frame_rate' in config:
            self.frame_rate = config['frame_rate']
    
    def initialize_tracker(self) -> bool:
        """
        Initialize ByteTrack tracker (or fallback to SimpleIOUTracker).
        
        Returns:
            True if successful, False otherwise
        """
        if SUPERVISION_AVAILABLE:
            try:
                logger.info(f"Initializing {self.tracker_type} tracker")
                
                self.tracker = BYTETracker(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=self.frame_rate
                )
                
                logger.info("ByteTrack initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize ByteTrack: {e}")
                logger.info("Falling back to SimpleIOUTracker")
        
        # Fallback to SimpleIOUTracker
        try:
            logger.info("Initializing SimpleIOUTracker as fallback")
            self.tracker = SimpleIOUTracker(
                iou_threshold=self.match_thresh,
                max_age=self.track_buffer
            )
            self.using_fallback = True
            logger.info("SimpleIOUTracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SimpleIOUTracker: {e}")
            return False
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries from PersonDetector
            frame: Optional frame for visualization (not used by ByteTrack)
            
        Returns:
            List of tracking dictionaries with track_id, bbox, confidence, etc.
        """
        if self.tracker is None:
            logger.error("Tracker not initialized")
            return []
        
        # Measure tracking time
        start_time = time.time()
        
        try:
            # Convert detections to supervision format
            supervision_dets = TrackingUtils.convert_detections_to_supervision(detections)
            
            # Update tracker
            if len(supervision_dets) > 0:
                tracked_detections = self.tracker.update(
                    supervision_dets,
                    [frame.shape[1], frame.shape[0]] if frame is not None else [640, 480]
                )
                track_ids = tracked_detections[:, 4].astype(int)
            else:
                tracked_detections = supervision_dets
                track_ids = np.array([])
            
            # Convert to standardized output
            tracking_output = TrackingUtils.convert_supervision_to_tracking_output(
                tracked_detections,
                track_ids
            )
            
            # Update track history
            self._update_track_history(tracking_output)
            
            # Update active tracks
            self._update_active_tracks(tracking_output)
            
            # Increment frame counter
            self._frame_number += 1
            
            # Record tracking time
            tracking_time = time.time() - start_time
            self._tracking_times.append(tracking_time)
            
            return tracking_output
            
        except Exception as e:
            logger.error(f"Tracking update failed: {e}")
            return []
    
    def _update_track_history(self, tracking_output: List[Dict[str, Any]]):
        """
        Update track history with new tracking results.
        
        Args:
            tracking_output: List of tracking dictionaries
        """
        for track in tracking_output:
            track_id = track.get('track_id')
            
            if track_id >= 0:  # Valid track ID
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                    self._total_tracks_created += 1
                
                # Add to history
                self._track_history[track_id].append({
                    'frame_number': self._frame_number,
                    'bbox': track.get('bbox'),
                    'confidence': track.get('confidence'),
                    'timestamp': time.time()
                })
    
    def _update_active_tracks(self, tracking_output: List[Dict[str, Any]]):
        """
        Update active tracks dictionary.
        
        Args:
            tracking_output: List of tracking dictionaries
        """
        # Clear previous active tracks
        self._active_tracks.clear()
        
        # Add current active tracks
        for track in tracking_output:
            track_id = track.get('track_id')
            
            if track_id >= 0:
                # Calculate track age
                age = TrackingUtils.calculate_track_age(
                    self._track_history.get(track_id, []),
                    self._frame_number
                )
                
                self._active_tracks[track_id] = {
                    'track_id': track_id,
                    'bbox': track.get('bbox'),
                    'confidence': track.get('confidence'),
                    'age': age,
                    'last_seen': self._frame_number
                }
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active tracks.
        
        Returns:
            List of active track dictionaries
        """
        return list(self._active_tracks.values())
    
    def get_track_history(self, track_id: int) -> List[Dict[str, Any]]:
        """
        Get history for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of track history entries
        """
        return self._track_history.get(track_id, [])
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking metrics
        """
        if not self._tracking_times:
            return {
                "tracking_count": 0,
                "avg_tracking_latency_ms": None,
                "median_tracking_latency_ms": None,
                "min_tracking_latency_ms": None,
                "max_tracking_latency_ms": None,
                "total_tracks_created": 0,
                "active_tracks": 0,
            }
        
        times_ms = [t * 1000 for t in self._tracking_times]
        
        return {
            "tracking_count": len(self._tracking_times),
            "avg_tracking_latency_ms": np.mean(times_ms),
            "median_tracking_latency_ms": np.median(times_ms),
            "min_tracking_latency_ms": np.min(times_ms),
            "max_tracking_latency_ms": np.max(times_ms),
            "total_tracks_created": self._total_tracks_created,
            "active_tracks": len(self._active_tracks),
        }
    
    def reset(self):
        """Reset tracker state."""
        self._track_id_counter = 0
        self._track_history.clear()
        self._active_tracks.clear()
        self._frame_number = 0
        self._tracking_times.clear()
        self._total_tracks_created = 0
        
        # Reinitialize tracker
        self.initialize_tracker()
        
        logger.info("Tracker reset")
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracking_output: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw tracking bounding boxes and IDs on frame.
        
        Args:
            frame: Input frame
            tracking_output: List of tracking dictionaries
            color: BGR color for bounding boxes
            thickness: Line thickness
            
        Returns:
            Frame with drawn tracks
        """
        frame_copy = frame.copy()
        
        for track in tracking_output:
            bbox = track.get('bbox', [])
            track_id = track.get('track_id', -1)
            
            if len(bbox) == 4 and track_id >= 0:
                # Draw bounding box
                cv2.rectangle(
                    frame_copy,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    thickness
                )
                
                # Draw track ID
                frame_copy = TrackingUtils.draw_track_id(
                    frame_copy, bbox, track_id, color, thickness
                )
        
        return frame_copy
    
    def __repr__(self) -> str:
        """String representation."""
        active = len(self._active_tracks)
        return f"ByteTrackManager(active_tracks={active}, total_created={self._total_tracks_created})"
