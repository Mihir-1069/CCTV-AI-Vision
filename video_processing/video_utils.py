"""
Video Utilities Module
Provides utility functions for video processing operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VideoUtils:
    """
    Utility functions for video processing.
    """
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video information without opening the video fully.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {}
        
        info = {
            "path": str(Path(video_path).absolute()),
            "filename": Path(video_path).name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": None,
        }
        
        # Calculate duration
        if info["fps"] and info["fps"] > 0:
            info["duration"] = info["total_frames"] / info["fps"]
        
        cap.release()
        return info
    
    @staticmethod
    def is_valid_video(video_path: str) -> bool:
        """
        Check if a file is a valid video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid video, False otherwise
        """
        path = Path(video_path)
        
        # Check file exists
        if not path.exists():
            return False
        
        # Check file extension
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if path.suffix.lower() not in supported_extensions:
            return False
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(str(path))
        is_valid = cap.isOpened()
        cap.release()
        
        return is_valid
    
    @staticmethod
    def create_synthetic_video(
        output_path: str,
        duration: float = 5.0,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        pattern: str = "gradient"
    ) -> bool:
        """
        Create a synthetic test video.
        
        Args:
            output_path: Path to save the video
            duration: Video duration in seconds
            fps: Frames per second
            width: Video width
            height: Video height
            pattern: Pattern type ('gradient', 'noise', 'moving_box')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            total_frames = int(duration * fps)
            
            for frame_num in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                if pattern == "gradient":
                    # Color gradient
                    for y in range(height):
                        for x in range(width):
                            frame[y, x] = [
                                int(255 * x / width),
                                int(255 * y / height),
                                int(255 * (x + y) / (width + height))
                            ]
                
                elif pattern == "noise":
                    # Random noise
                    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                elif pattern == "moving_box":
                    # Moving box
                    frame[:] = [50, 50, 50]  # Background
                    box_x = int((frame_num * 10) % (width - 100))
                    box_y = int((frame_num * 5) % (height - 100))
                    cv2.rectangle(frame, (box_x, box_y), (box_x + 100, box_y + 100), (0, 255, 0), -1)
                    
                    # Add frame number
                    cv2.putText(
                        frame, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                
                out.write(frame)
            
            out.release()
            logger.info(f"Synthetic video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create synthetic video: {e}")
            return False
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str) -> bool:
        """
        Save a single frame as an image.
        
        Args:
            frame: Frame to save
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, frame)
            logger.debug(f"Frame saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
    
    @staticmethod
    def calculate_aspect_ratio(width: int, height: int) -> str:
        """
        Calculate aspect ratio as string.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Aspect ratio string (e.g., "16:9")
        """
        from math import gcd
        
        divisor = gcd(width, height)
        w_ratio = width // divisor
        h_ratio = height // divisor
        
        return f"{w_ratio}:{h_ratio}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to HH:MM:SS.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
