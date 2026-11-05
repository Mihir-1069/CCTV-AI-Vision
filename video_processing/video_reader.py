"""
Video Reader Module
Provides reusable VideoReader class for reading and processing video files.
"""

import cv2
import os
from typing import Optional, Tuple, Generator, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """
    A reusable video reader class for processing video files.
    
    Supports:
    - MP4, AVI, MOV, MKV, WEBM formats
    - Frame extraction with metadata
    - Context manager usage
    - Error handling for corrupted videos
    """
    
    def __init__(self, video_path: str):
        """
        Initialize VideoReader.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If file is not a valid video
        """
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        
        # Video metadata
        self._fps: Optional[float] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._total_frames: Optional[int] = None
        self._duration: Optional[float] = None
        
        # Validate file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate file extension
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if self.video_path.suffix.lower() not in supported_extensions:
            logger.warning(
                f"File extension '{self.video_path.suffix}' may not be supported. "
                f"Supported: {supported_extensions}"
            )
    
    def open(self) -> bool:
        """
        Open the video file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {self.video_path}")
                return False
            
            # Read metadata
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            if self._fps and self._fps > 0:
                self._duration = self._total_frames / self._fps
            else:
                self._duration = None
            
            self._is_open = True
            logger.info(
                f"Video opened: {self.video_path.name} | "
                f"Resolution: {self._width}x{self._height} | "
                f"FPS: {self._fps:.2f} | "
                f"Frames: {self._total_frames} | "
                f"Duration: {self._duration:.2f}s"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            self.close()
            return False
    
    def close(self) -> None:
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_open = False
        logger.debug(f"Video closed: {self.video_path}")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def read_frame(self) -> Tuple[Optional[cv2.typing.MatLike], int, Optional[float]]:
        """
        Read a single frame from the video.
        
        Returns:
            Tuple of (frame, frame_number, timestamp)
            frame: numpy array or None if failed
            frame_number: current frame index
            timestamp: timestamp in seconds or None if FPS unavailable
        """
        if not self._is_open or self.cap is None:
            logger.error("Video not opened")
            return None, -1, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            logger.debug("End of video or failed to read frame")
            return None, -1, None
        
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = None
        
        if self._fps and self._fps > 0:
            timestamp = frame_number / self._fps
        
        return frame, frame_number, timestamp
    
    def get_frame_at(self, frame_number: int) -> Optional[cv2.typing.MatLike]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_number: Frame index to retrieve
            
        Returns:
            Frame as numpy array or None if failed
        """
        if not self._is_open or self.cap is None:
            logger.error("Video not opened")
            return None
        
        if frame_number < 0 or frame_number >= self.total_frames:
            logger.error(f"Frame number out of range: {frame_number}")
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            return frame
        
        return None
    
    def frames(self, skip_frames: int = 0) -> Generator[Tuple[cv2.typing.MatLike, int, Optional[float]], None, None]:
        """
        Generator that yields frames sequentially.
        
        Args:
            skip_frames: Number of frames to skip between yields (0 = all frames)
            
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        frame_count = 0
        
        while True:
            frame, frame_number, timestamp = self.read_frame()
            
            if frame is None:
                break
            
            # Apply frame skipping
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            yield frame, frame_number, timestamp
            frame_count += 1
    
    @property
    def fps(self) -> Optional[float]:
        """Video frame rate."""
        return self._fps
    
    @property
    def width(self) -> Optional[int]:
        """Video width in pixels."""
        return self._width
    
    @property
    def height(self) -> Optional[int]:
        """Video height in pixels."""
        return self._height
    
    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        """Video resolution as (width, height)."""
        if self._width and self._height:
            return (self._width, self._height)
        return None
    
    @property
    def total_frames(self) -> Optional[int]:
        """Total number of frames in video."""
        return self._total_frames
    
    @property
    def duration(self) -> Optional[float]:
        """Video duration in seconds."""
        return self._duration
    
    @property
    def is_open(self) -> bool:
        """Whether video is currently open."""
        return self._is_open
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get video metadata as dictionary.
        
        Returns:
            Dictionary containing video metadata
        """
        return {
            "path": str(self.video_path),
            "filename": self.video_path.name,
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
            "resolution": self.resolution,
            "total_frames": self._total_frames,
            "duration": self._duration,
            "is_open": self._is_open,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "open" if self._is_open else "closed"
        return f"VideoReader({self.video_path.name}, {status})"
