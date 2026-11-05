"""
Unit tests for FrameProcessor module.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.frame_processor import FrameProcessor


class TestFrameProcessor:
    """Test cases for FrameProcessor class."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def processor(self):
        """Create a FrameProcessor instance."""
        return FrameProcessor(
            target_width=320,
            target_height=240,
            maintain_aspect_ratio=True,
            frame_skip=0
        )
    
    def test_frame_processor_init(self):
        """Test FrameProcessor initialization."""
        processor = FrameProcessor(
            target_width=640,
            target_height=480,
            maintain_aspect_ratio=True,
            frame_skip=2
        )
        
        assert processor.target_width == 640
        assert processor.target_height == 480
        assert processor.maintain_aspect_ratio is True
        assert processor.frame_skip == 2
    
    def test_resize_frame_no_resize(self, sample_frame):
        """Test resize with no target dimensions."""
        processor = FrameProcessor(target_width=None, target_height=None)
        resized = processor.resize_frame(sample_frame)
        
        assert resized.shape == sample_frame.shape
        assert np.array_equal(resized, sample_frame)
    
    def test_resize_frame_with_dimensions(self, sample_frame, processor):
        """Test resize with target dimensions."""
        resized = processor.resize_frame(sample_frame)
        
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320
    
    def test_resize_frame_aspect_ratio(self, sample_frame):
        """Test resize maintaining aspect ratio."""
        processor = FrameProcessor(
            target_width=320,
            target_height=240,
            maintain_aspect_ratio=True
        )
        
        resized = processor.resize_frame(sample_frame)
        
        # Should maintain aspect ratio and pad if needed
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320
    
    def test_resize_frame_no_aspect_ratio(self, sample_frame):
        """Test resize without maintaining aspect ratio."""
        processor = FrameProcessor(
            target_width=320,
            target_height=240,
            maintain_aspect_ratio=False
        )
        
        resized = processor.resize_frame(sample_frame)
        
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320
    
    def test_resize_frame_width_only(self, sample_frame):
        """Test resize with only width specified."""
        processor = FrameProcessor(target_width=320, target_height=None)
        resized = processor.resize_frame(sample_frame)
        
        assert resized.shape[1] == 320
        # Height should be calculated based on aspect ratio
        expected_height = int(480 * (320 / 640))
        assert resized.shape[0] == expected_height
    
    def test_resize_frame_height_only(self, sample_frame):
        """Test resize with only height specified."""
        processor = FrameProcessor(target_width=None, target_height=240)
        resized = processor.resize_frame(sample_frame)
        
        assert resized.shape[0] == 240
        # Width should be calculated based on aspect ratio
        expected_width = int(640 * (240 / 480))
        assert resized.shape[1] == expected_width
    
    def test_resize_frame_override(self, sample_frame, processor):
        """Test resize with override dimensions."""
        resized = processor.resize_frame(
            sample_frame,
            target_width=160,
            target_height=120
        )
        
        assert resized.shape[0] == 120
        assert resized.shape[1] == 160
    
    def test_resize_frame_none_input(self):
        """Test resize with None frame."""
        processor = FrameProcessor(target_width=320, target_height=240)
        
        with pytest.raises(ValueError):
            processor.resize_frame(None)
    
    def test_validate_frame_valid(self, sample_frame):
        """Test frame validation with valid frame."""
        processor = FrameProcessor()
        assert processor.validate_frame(sample_frame) is True
    
    def test_validate_frame_none(self):
        """Test frame validation with None."""
        processor = FrameProcessor()
        assert processor.validate_frame(None) is False
    
    def test_validate_frame_not_array(self):
        """Test frame validation with non-array."""
        processor = FrameProcessor()
        assert processor.validate_frame("not_an_array") is False
    
    def test_validate_frame_empty(self):
        """Test frame validation with empty array."""
        processor = FrameProcessor()
        empty_frame = np.array([])
        assert processor.validate_frame(empty_frame) is False
    
    def test_should_process_frame_no_skip(self):
        """Test frame processing with no skip."""
        processor = FrameProcessor(frame_skip=0)
        
        assert processor.should_process_frame(0) is True
        assert processor.should_process_frame(5) is True
        assert processor.should_process_frame(10) is True
    
    def test_should_process_frame_with_skip(self):
        """Test frame processing with skip."""
        processor = FrameProcessor(frame_skip=2)
        
        assert processor.should_process_frame(0) is True
        assert processor.should_process_frame(1) is False
        assert processor.should_process_frame(2) is False
        assert processor.should_process_frame(3) is True
        assert processor.should_process_frame(6) is True
    
    def test_preprocess_frame_no_ops(self, sample_frame):
        """Test preprocessing with no operations."""
        processor = FrameProcessor()
        processed = processor.preprocess_frame(sample_frame)
        
        assert np.array_equal(processed, sample_frame)
    
    def test_preprocess_frame_normalize(self, sample_frame):
        """Test preprocessing with normalization."""
        processor = FrameProcessor()
        processed = processor.preprocess_frame(sample_frame, normalize=True)
        
        assert processed.dtype == np.float32
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
    
    def test_preprocess_frame_bgr2rgb(self, sample_frame):
        """Test preprocessing with BGR to RGB conversion."""
        processor = FrameProcessor()
        processed = processor.preprocess_frame(sample_frame, convert_color='BGR2RGB')
        
        # Check that channels are swapped
        assert not np.array_equal(processed, sample_frame)
        # Convert back should match original
        original_back = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        assert np.array_equal(original_back, sample_frame)
    
    def test_preprocess_frame_bgr2gray(self, sample_frame):
        """Test preprocessing with BGR to grayscale conversion."""
        processor = FrameProcessor()
        processed = processor.preprocess_frame(sample_frame, convert_color='BGR2GRAY')
        
        assert len(processed.shape) == 2  # Grayscale
    
    def test_preprocess_frame_invalid_conversion(self, sample_frame):
        """Test preprocessing with invalid color conversion."""
        processor = FrameProcessor()
        processed = processor.preprocess_frame(sample_frame, convert_color='INVALID')
        
        # Should return original frame with warning
        assert np.array_equal(processed, sample_frame)
    
    def test_preprocess_frame_invalid_frame(self):
        """Test preprocessing with invalid frame."""
        processor = FrameProcessor()
        
        with pytest.raises(ValueError):
            processor.preprocess_frame(None)
    
    def test_calculate_frame_timestamp(self):
        """Test timestamp calculation."""
        processor = FrameProcessor()
        
        assert processor.calculate_frame_timestamp(0, 30) == 0.0
        assert processor.calculate_frame_timestamp(30, 30) == 1.0
        assert processor.calculate_frame_timestamp(60, 30) == 2.0
        assert processor.calculate_frame_timestamp(45, 30) == 1.5
    
    def test_calculate_frame_timestamp_invalid_fps(self):
        """Test timestamp calculation with invalid FPS."""
        processor = FrameProcessor()
        
        assert processor.calculate_frame_timestamp(30, 0) == 0.0
        assert processor.calculate_frame_timestamp(30, -10) == 0.0
    
    def test_get_frame_info(self, sample_frame):
        """Test getting frame information."""
        processor = FrameProcessor()
        info = processor.get_frame_info(sample_frame, 10, 30.0)
        
        assert info["frame_number"] == 10
        assert info["timestamp"] == 10 / 30.0
        assert info["width"] == 640
        assert info["height"] == 480
        assert info["channels"] == 3
        assert info["dtype"] == "uint8"
        assert info["shape"] == (480, 640, 3)
    
    def test_get_frame_info_invalid_frame(self):
        """Test getting frame info with invalid frame."""
        processor = FrameProcessor()
        info = processor.get_frame_info(None, 0, 30.0)
        
        assert info == {}
