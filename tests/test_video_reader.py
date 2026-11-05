"""
Unit tests for VideoReader module.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.video_reader import VideoReader
from video_processing.video_utils import VideoUtils


class TestVideoReader:
    """Test cases for VideoReader class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def synthetic_video_path(self, temp_dir):
        """Create a synthetic test video."""
        video_path = temp_dir / "test_video.mp4"
        VideoUtils.create_synthetic_video(
            str(video_path),
            duration=2.0,
            fps=30,
            width=640,
            height=480,
            pattern="moving_box"
        )
        return str(video_path)
    
    def test_video_reader_init_valid_path(self, synthetic_video_path):
        """Test VideoReader initialization with valid path."""
        reader = VideoReader(synthetic_video_path)
        assert reader.video_path == Path(synthetic_video_path)
        assert not reader.is_open
    
    def test_video_reader_init_invalid_path(self):
        """Test VideoReader initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            VideoReader("nonexistent_video.mp4")
    
    def test_video_reader_open(self, synthetic_video_path):
        """Test opening a video file."""
        reader = VideoReader(synthetic_video_path)
        assert reader.open() is True
        assert reader.is_open
        reader.close()
    
    def test_video_reader_context_manager(self, synthetic_video_path):
        """Test VideoReader as context manager."""
        with VideoReader(synthetic_video_path) as reader:
            assert reader.is_open
        assert not reader.is_open
    
    def test_video_reader_metadata(self, synthetic_video_path):
        """Test reading video metadata."""
        with VideoReader(synthetic_video_path) as reader:
            assert reader.fps == 30.0
            assert reader.width == 640
            assert reader.height == 480
            assert reader.resolution == (640, 480)
            assert reader.total_frames == 60  # 2.0s * 30fps
            assert reader.duration == 2.0
    
    def test_video_reader_read_frame(self, synthetic_video_path):
        """Test reading a single frame."""
        with VideoReader(synthetic_video_path) as reader:
            frame, frame_number, timestamp = reader.read_frame()
            
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame_number is not None
            assert timestamp is not None
            assert frame.shape == (480, 640, 3)
    
    def test_video_reader_get_frame_at(self, synthetic_video_path):
        """Test getting a specific frame."""
        with VideoReader(synthetic_video_path) as reader:
            frame = reader.get_frame_at(30)
            
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
    
    def test_video_reader_get_frame_at_invalid(self, synthetic_video_path):
        """Test getting frame with invalid index."""
        with VideoReader(synthetic_video_path) as reader:
            frame = reader.get_frame_at(1000)  # Out of range
            assert frame is None
    
    def test_video_reader_frames_generator(self, synthetic_video_path):
        """Test frames generator."""
        with VideoReader(synthetic_video_path) as reader:
            frame_count = 0
            for frame, frame_number, timestamp in reader.frames():
                assert frame is not None
                assert isinstance(frame, np.ndarray)
                assert frame_number >= frame_count
                frame_count += 1
            
            assert frame_count == 60
    
    def test_video_reader_frames_with_skip(self, synthetic_video_path):
        """Test frames generator with skipping."""
        with VideoReader(synthetic_video_path) as reader:
            frame_count = 0
            for frame, frame_number, timestamp in reader.frames(skip_frames=4):
                frame_count += 1
            
            # Should process every 5th frame: 60 / 5 = 12 frames
            assert frame_count == 12
    
    def test_video_reader_get_metadata_dict(self, synthetic_video_path):
        """Test getting metadata as dictionary."""
        with VideoReader(synthetic_video_path) as reader:
            metadata = reader.get_metadata()
            
            assert metadata["filename"] == Path(synthetic_video_path).name
            assert metadata["fps"] == 30.0
            assert metadata["width"] == 640
            assert metadata["height"] == 480
            assert metadata["total_frames"] == 60
            assert metadata["duration"] == 2.0
            assert metadata["is_open"] is True
    
    def test_video_reader_close(self, synthetic_video_path):
        """Test closing video reader."""
        reader = VideoReader(synthetic_video_path)
        reader.open()
        assert reader.is_open
        
        reader.close()
        assert not reader.is_open


class TestVideoUtils:
    """Test cases for VideoUtils class."""
    
    @pytest.fixture
    def temp_dir_2(self):
        """Create a temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def synthetic_video_path_2(self, temp_dir_2):
        """Create a synthetic test video."""
        video_path = temp_dir_2 / "test_video.mp4"
        VideoUtils.create_synthetic_video(
            str(video_path),
            duration=1.0,
            fps=30,
            width=640,
            height=480,
            pattern="gradient"
        )
        return str(video_path)
    
    def test_create_synthetic_video(self, temp_dir_2):
        """Test creating synthetic video."""
        video_path = temp_dir_2 / "synthetic.mp4"
        result = VideoUtils.create_synthetic_video(
            str(video_path),
            duration=1.0,
            fps=30,
            width=320,
            height=240
        )
        
        assert result is True
        assert video_path.exists()
    
    def test_get_video_info(self, synthetic_video_path_2):
        """Test getting video information."""
        info = VideoUtils.get_video_info(synthetic_video_path_2)
        
        assert info["fps"] == 30.0
        assert info["width"] == 640
        assert info["height"] == 480
        assert info["total_frames"] == 30
        assert info["duration"] == 1.0
    
    def test_is_valid_video(self, synthetic_video_path_2):
        """Test video validation."""
        assert VideoUtils.is_valid_video(synthetic_video_path_2) is True
        assert VideoUtils.is_valid_video("nonexistent.mp4") is False
    
    def test_save_frame(self, temp_dir_2, synthetic_video_path_2):
        """Test saving a frame as image."""
        with VideoReader(synthetic_video_path_2) as reader:
            frame, _, _ = reader.read_frame()
        
        output_path = temp_dir_2 / "test_frame.jpg"
        result = VideoUtils.save_frame(frame, str(output_path))
        
        assert result is True
        assert output_path.exists()
    
    def test_calculate_aspect_ratio(self):
        """Test aspect ratio calculation."""
        assert VideoUtils.calculate_aspect_ratio(1920, 1080) == "16:9"
        assert VideoUtils.calculate_aspect_ratio(1280, 720) == "16:9"
        assert VideoUtils.calculate_aspect_ratio(640, 480) == "4:3"
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert VideoUtils.format_duration(90) == "01:30"
        assert VideoUtils.format_duration(3665) == "01:01:05"
        assert VideoUtils.format_duration(45) == "00:45"
