"""
AI Engine Module
SchoolVision AI - Computer Vision Pipeline
Person 1 - Computer Vision and AI Pipeline Development
"""

from .detection.yolo_detector import PersonDetector
from .tracking.bytetrack_manager import ByteTrackManager
from .analytics.occupancy_manager import OccupancyManager
from .analytics.entry_exit_counter import EntryExitCounter
from .analytics.zone_manager import ZoneManager
from .pose.pose_estimator import PoseEstimator
from .activity.activity_recognition import ActivityRecognizer, ActivityLabel, ActivityResult
from .anomaly.autoencoder import AnomalyDetector
from .crowd.crowd_analytics import CrowdAnalytics

__all__ = [
    "PersonDetector",
    "ByteTrackManager",
    "OccupancyManager",
    "EntryExitCounter",
    "ZoneManager",
    "PoseEstimator",
    "ActivityRecognizer",
    "ActivityLabel",
    "ActivityResult",
    "AnomalyDetector",
    "CrowdAnalytics",
]
