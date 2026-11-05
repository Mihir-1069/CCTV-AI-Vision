"""
Detection Utilities Module
Provides utility functions for YOLO detection operations.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DetectionUtils:
    """
    Utility functions for detection operations.
    """
    
    # COCO class names
    COCO_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }
    
    @staticmethod
    def get_class_name(class_id: int) -> str:
        """
        Get class name from COCO class ID.
        
        Args:
            class_id: COCO class ID
            
        Returns:
            Class name or 'unknown' if not found
        """
        return DetectionUtils.COCO_CLASSES.get(class_id, 'unknown')
    
    @staticmethod
    def validate_bbox(bbox: List[int], image_width: int, image_height: int) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            image_width: Image width
            image_height: Image height
            
        Returns:
            True if valid, False otherwise
        """
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Check coordinates are within bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            return False
        
        # Check x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
    
    @staticmethod
    def clip_bbox(bbox: List[int], image_width: int, image_height: int) -> List[int]:
        """
        Clip bounding box to image boundaries.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            image_width: Image width
            image_height: Image height
            
        Returns:
            Clipped bounding box
        """
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        
        return [x1, y1, x2, y2]
    
    @staticmethod
    def draw_bbox(
        frame: np.ndarray,
        bbox: List[int],
        label: str,
        confidence: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box in [x1, y1, x2, y2] format
            label: Class label
            confidence: Detection confidence
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Frame with drawn bounding box
        """
        frame_copy = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(
            frame_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame_copy,
            label_text,
            (x1 + 5, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        return frame_copy
    
    @staticmethod
    def filter_detections_by_class(
        detections: List[Dict[str, Any]],
        target_classes: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Filter detections by target class IDs.
        
        Args:
            detections: List of detection dictionaries
            target_classes: List of target class IDs
            
        Returns:
            Filtered detections
        """
        return [
            det for det in detections
            if det.get('class_id') in target_classes
        ]
    
    @staticmethod
    def filter_detections_by_confidence(
        detections: List[Dict[str, Any]],
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection dictionaries
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Filtered detections
        """
        return [
            det for det in detections
            if det.get('confidence', 0) >= confidence_threshold
        ]
    
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
