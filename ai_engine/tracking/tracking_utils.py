"""
Tracking Utilities Module
Provides utility functions for tracking operations.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TrackingUtils:
    """
    Utility functions for tracking operations.
    """
    
    @staticmethod
    def convert_detections_to_supervision(
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Convert detection format to supervision format.
        
        Args:
            detections: List of detection dictionaries with bbox and confidence
            
        Returns:
            numpy array of shape (N, 6) with [x1, y1, x2, y2, confidence, class_id]
        """
        if not detections:
            return np.empty((0, 6))
        
        supervision_detections = []
        
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            
            supervision_detections.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                confidence, class_id
            ])
        
        return np.array(supervision_detections, dtype=np.float32)
    
    @staticmethod
    def convert_supervision_to_tracking_output(
        detections: np.ndarray,
        track_ids: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Convert supervision tracking results to standardized output.
        
        Args:
            detections: numpy array of shape (N, 6) with [x1, y1, x2, y2, confidence, class_id]
            track_ids: numpy array of track IDs
            
        Returns:
            List of tracking dictionaries
        """
        if len(detections) == 0:
            return []
        
        tracking_output = []
        
        for i in range(len(detections)):
            x1, y1, x2, y2, confidence, class_id = detections[i]
            track_id = int(track_ids[i]) if i < len(track_ids) else -1
            
            tracking_output.append({
                "track_id": track_id,
                "class_id": int(class_id),
                "class_name": "person" if int(class_id) == 0 else "unknown",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidence),
                "track_state": "tracked" if track_id >= 0 else "lost"
            })
        
        return tracking_output
    
    @staticmethod
    def draw_track_id(
        frame: np.ndarray,
        bbox: List[int],
        track_id: int,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw track ID on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID
            color: BGR color
            thickness: Line thickness
            
        Returns:
            Frame with drawn track ID
        """
        frame_copy = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw track ID above bbox
        label_text = f"ID: {track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background rectangle
        cv2.rectangle(
            frame_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame_copy,
            label_text,
            (x1 + 5, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame_copy
    
    @staticmethod
    def draw_tracking_info(
        frame: np.ndarray,
        active_tracks: int,
        frame_number: int,
        detection_latency: float,
        tracking_latency: float
    ) -> np.ndarray:
        """
        Draw tracking information overlay on frame.
        
        Args:
            frame: Input frame
            active_tracks: Number of active tracks
            frame_number: Current frame number
            detection_latency: Detection latency in ms
            tracking_latency: Tracking latency in ms
            
        Returns:
            Frame with tracking info overlay
        """
        frame_copy = frame.copy()
        
        info_lines = [
            f"Frame: {frame_number}",
            f"Active Tracks: {active_tracks}",
            f"Det Latency: {detection_latency:.1f}ms",
            f"Track Latency: {tracking_latency:.1f}ms"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame_copy,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25
        
        return frame_copy
    
    @staticmethod
    def calculate_track_age(
        track_history: List[Dict[str, Any]],
        current_frame: int
    ) -> int:
        """
        Calculate track age in frames.
        
        Args:
            track_history: List of track history entries
            current_frame: Current frame number
            
        Returns:
            Track age in frames
        """
        if not track_history:
            return 0
        
        first_frame = min(entry.get('frame_number', current_frame) for entry in track_history)
        return current_frame - first_frame
    
    @staticmethod
    def filter_tracks_by_age(
        tracks: List[Dict[str, Any]],
        min_age: int,
        max_age: int
    ) -> List[Dict[str, Any]]:
        """
        Filter tracks by age.
        
        Args:
            tracks: List of track dictionaries
            min_age: Minimum age in frames
            max_age: Maximum age in frames
            
        Returns:
            Filtered tracks
        """
        filtered = []
        for track in tracks:
            age = track.get('age', 0)
            if min_age <= age <= max_age:
                filtered.append(track)
        return filtered
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
