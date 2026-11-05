"""
Video Processing Module
SchoolVision AI - Computer Vision Pipeline
Person 1 - Computer Vision and AI Pipeline Development
"""

from .video_reader import VideoReader
from .frame_processor import FrameProcessor
from .video_utils import VideoUtils

__all__ = [
    "VideoReader",
    "FrameProcessor",
    "VideoUtils",
]
