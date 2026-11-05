"""
Detection Module
SchoolVision AI - Computer Vision Pipeline
Person 1 - Computer Vision and AI Pipeline Development
"""

from .yolo_detector import PersonDetector
from .detection_utils import DetectionUtils

__all__ = [
    "PersonDetector",
    "DetectionUtils",
]
