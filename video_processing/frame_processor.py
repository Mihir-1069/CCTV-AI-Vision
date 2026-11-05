"""
Frame Processor Module
Provides frame processing utilities for resizing, validation, and preprocessing.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Frame processing utilities for video frames.
    
    Supports:
    - Configurable frame resizing
    - Frame validation
    - Frame skipping logic
    - CPU-friendly processing
    """
    
    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        maintain_aspect_ratio: bool = True,
        frame_skip: int = 0
    ):
        """
        Initialize FrameProcessor.
        
        Args:
            target_width: Target width for resizing (None = keep original)
            target_height: Target height for resizing (None = keep original)
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            frame_skip: Number of frames to skip (0 = process all frames)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.frame_skip = frame_skip
        
        # Calculate scaling factor if both dimensions specified
        self._scale_factor = None
        if target_width and target_height:
            self._scale_factor = (target_width, target_height)
    
    def resize_frame(
        self,
        frame: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ) -> np.ndarray:
        """
        Resize a frame to target dimensions.
        
        Args:
            frame: Input frame as numpy array
            target_width: Override target width (optional)
            target_height: Override target height (optional)
            
        Returns:
            Resized frame
        """
        if frame is None:
            raise ValueError("Frame cannot be None")
        
        # Use override values if provided
        width = target_width if target_width is not None else self.target_width
        height = target_height if target_height is not None else self.target_height
        
        # If no resizing needed, return original
        if width is None and height is None:
            return frame
        
        original_height, original_width = frame.shape[:2]
        
        # If only one dimension specified, calculate the other maintaining aspect ratio
        if width and not height:
            aspect_ratio = original_height / original_width
            height = int(width * aspect_ratio)
        elif height and not width:
            aspect_ratio = original_width / original_height
            width = int(height * aspect_ratio)
        
        # Resize frame
        if self.maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(width / original_width, height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target dimensions if needed
            if new_width != width or new_height != height:
                pad_width = width - new_width
                pad_height = height - new_height
                top = pad_height // 2
                bottom = pad_height - top
                left = pad_width // 2
                right = pad_width - left
                resized = cv2.copyMakeBorder(
                    resized, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
        else:
            # Direct resize without aspect ratio
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate a frame.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if frame is None:
            logger.warning("Frame is None")
            return False
        
        if not isinstance(frame, np.ndarray):
            logger.warning("Frame is not a numpy array")
            return False
        
        if len(frame.shape) < 2:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return False
        
        if frame.size == 0:
            logger.warning("Frame is empty")
            return False
        
        return True
    
    def should_process_frame(self, frame_number: int) -> bool:
        """
        Determine if a frame should be processed based on skip logic.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            True if frame should be processed, False otherwise
        """
        if self.frame_skip == 0:
            return True
        
        return frame_number % (self.frame_skip + 1) == 0
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        normalize: bool = False,
        convert_color: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply preprocessing to a frame.
        
        Args:
            frame: Input frame
            normalize: Whether to normalize pixel values to [0, 1]
            convert_color: Color space conversion (e.g., 'BGR2RGB', 'BGR2GRAY')
            
        Returns:
            Preprocessed frame
        """
        if not self.validate_frame(frame):
            raise ValueError("Invalid frame")
        
        processed = frame.copy()
        
        # Color space conversion
        if convert_color:
            conversion_map = {
                'BGR2RGB': cv2.COLOR_BGR2RGB,
                'BGR2GRAY': cv2.COLOR_BGR2GRAY,
                'RGB2BGR': cv2.COLOR_RGB2BGR,
                'RGB2GRAY': cv2.COLOR_RGB2GRAY,
            }
            
            if convert_color in conversion_map:
                processed = cv2.cvtColor(processed, conversion_map[convert_color])
            else:
                logger.warning(f"Unknown color conversion: {convert_color}")
        
        # Normalization
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def calculate_frame_timestamp(
        self,
        frame_number: int,
        fps: float
    ) -> float:
        """
        Calculate timestamp for a frame.
        
        Args:
            frame_number: Frame number
            fps: Video frame rate
            
        Returns:
            Timestamp in seconds
        """
        if fps <= 0:
            logger.warning(f"Invalid FPS: {fps}")
            return 0.0
        
        return frame_number / fps
    
    def get_frame_info(self, frame: np.ndarray, frame_number: int, fps: float) -> dict:
        """
        Get information about a frame.
        
        Args:
            frame: Frame array
            frame_number: Frame number
            fps: Video FPS
            
        Returns:
            Dictionary with frame information
        """
        if not self.validate_frame(frame):
            return {}
        
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        timestamp = self.calculate_frame_timestamp(frame_number, fps)
        
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(frame.dtype),
            "shape": frame.shape,
        }
