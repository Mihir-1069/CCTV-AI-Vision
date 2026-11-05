"""
Demo script for testing video processing module.
Accepts a video path through command-line arguments and demonstrates video processing capabilities.
"""

import argparse
import sys
import time
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.video_reader import VideoReader
from video_processing.frame_processor import FrameProcessor
from video_processing.video_utils import VideoUtils
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Test video processing module"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Target width for resizing"
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Target height for resizing"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip (0 = process all)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display processed frames"
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Path to save processed video"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Use config values if not specified in arguments
    if args.resize_width is None and config:
        args.resize_width = config.get("performance", {}).get("resize_width")
    if args.resize_height is None and config:
        args.resize_height = config.get("performance", {}).get("resize_height")
    if args.frame_skip == 0 and config:
        args.frame_skip = config.get("performance", {}).get("frame_skip", 0)
    
    # Validate video path
    if not Path(args.source).exists():
        logger.error(f"Video file not found: {args.source}")
        return 1
    
    logger.info(f"Starting video processing demo")
    logger.info(f"Source: {args.source}")
    logger.info(f"Resize: {args.resize_width}x{args.resize_height}")
    logger.info(f"Frame skip: {args.frame_skip}")
    
    # Initialize frame processor
    processor = FrameProcessor(
        target_width=args.resize_width,
        target_height=args.resize_height,
        maintain_aspect_ratio=True,
        frame_skip=args.frame_skip
    )
    
    # Performance tracking
    start_time = time.time()
    total_frames = 0
    processed_frames = 0
    
    try:
        # Open video
        with VideoReader(args.source) as reader:
            # Display metadata
            metadata = reader.get_metadata()
            logger.info("=" * 60)
            logger.info("VIDEO METADATA")
            logger.info("=" * 60)
            logger.info(f"Filename: {metadata['filename']}")
            logger.info(f"Resolution: {metadata['width']}x{metadata['height']}")
            logger.info(f"FPS: {metadata['fps']:.2f}")
            logger.info(f"Total Frames: {metadata['total_frames']}")
            logger.info(f"Duration: {metadata['duration']:.2f}s")
            logger.info("=" * 60)
            
            # Process frames
            logger.info("Processing frames...")
            
            for frame, frame_number, timestamp in reader.frames(skip_frames=args.frame_skip):
                total_frames += 1
                
                # Check max frames limit
                if args.max_frames and processed_frames >= args.max_frames:
                    logger.info(f"Reached max frames limit: {args.max_frames}")
                    break
                
                # Validate frame
                if not processor.validate_frame(frame):
                    logger.warning(f"Invalid frame at {frame_number}")
                    continue
                
                # Resize frame if needed
                if args.resize_width or args.resize_height:
                    frame = processor.resize_frame(frame)
                
                processed_frames += 1
                
                # Display frame if enabled
                if args.display:
                    cv2.imshow('Video Processing Demo', frame)
                    
                    # Add frame info overlay
                    info_text = f"Frame: {frame_number} | Time: {timestamp:.2f}s"
                    cv2.putText(
                        frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break
                
                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
            
            # Clean up display
            if args.display:
                cv2.destroyAllWindows()
        
        # Calculate performance
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames read: {total_frames}")
        logger.info(f"Frames processed: {processed_frames}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        if processing_time > 0:
            effective_fps = processed_frames / processing_time
            logger.info(f"Effective processing FPS: {effective_fps:.2f}")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import cv2
    sys.exit(main())
