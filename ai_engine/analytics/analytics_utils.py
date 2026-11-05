"""
Analytics Utilities Module
Provides utility functions for occupancy and entry/exit analytics.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalyticsUtils:
    """
    Utility functions for analytics operations.
    """
    
    @staticmethod
    def calculate_bbox_center(bbox: List[int]) -> Tuple[float, float]:
        """
        Calculate the center point of a bounding box.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            
        Returns:
            Tuple of (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)
    
    @staticmethod
    def point_to_line_side(
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> float:
        """
        Determine which side of a line a point is on using cross product.
        
        Args:
            point: Point coordinates (x, y)
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
            
        Returns:
            Positive if on one side, negative on other, zero on line
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Cross product: (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross_product
    
    @staticmethod
    def detect_line_crossing(
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> str:
        """
        Detect if a line was crossed between two points.
        
        Args:
            prev_point: Previous position (x, y)
            curr_point: Current position (x, y)
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
            
        Returns:
            "entry", "exit", or "none"
        """
        prev_side = AnalyticsUtils.point_to_line_side(prev_point, line_start, line_end)
        curr_side = AnalyticsUtils.point_to_line_side(curr_point, line_start, line_end)
        
        # If both on same side, no crossing
        if (prev_side > 0 and curr_side > 0) or (prev_side < 0 and curr_side < 0):
            return "none"
        
        # If one on line, no crossing
        if prev_side == 0 or curr_side == 0:
            return "none"
        
        # Crossing detected - determine direction
        if prev_side < 0 and curr_side > 0:
            return "entry"
        elif prev_side > 0 and curr_side < 0:
            return "exit"
        
        return "none"
    
    @staticmethod
    def point_in_polygon(
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]]
    ) -> bool:
        """
        Determine if a point is inside a polygon using ray casting algorithm.
        
        Args:
            point: Point coordinates (x, y)
            polygon: List of polygon vertices [(x, y), ...]
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def format_timestamp(dt: datetime = None) -> str:
        """
        Format timestamp as ISO 8601 string.
        
        Args:
            dt: datetime object (uses current time if None)
            
        Returns:
            ISO 8601 formatted timestamp string
        """
        if dt is None:
            dt = datetime.now()
        return dt.isoformat()
    
    @staticmethod
    def validate_analytics_output(output: Dict[str, Any]) -> bool:
        """
        Validate analytics output structure.
        
        Args:
            output: Analytics output dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["camera_id", "timestamp", "frame_number", "occupancy"]
        
        for key in required_keys:
            if key not in output:
                return False
        
        # Validate occupancy structure
        if "occupancy" in output:
            occupancy = output["occupancy"]
            if not isinstance(occupancy, dict):
                return False
            if "current_count" not in occupancy:
                return False
        
        return True
    
    @staticmethod
    def calculate_distance(
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def point_line_distance(
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> float:
        """
        Calculate perpendicular distance from point to line.
        
        Args:
            point: Point coordinates (x, y)
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
            
        Returns:
            Perpendicular distance
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line length
        line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_len == 0:
            return AnalyticsUtils.calculate_distance(point, line_start)
        
        # Calculate distance using cross product
        numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        distance = numerator / line_len
        
        return distance
