"""
Demo script for testing YOLO person detection.
Accepts a video path and performs person detection using YOLO11n.
"""

import argparse
import sys
import time
import json
from pathlib import Path
import logging
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.video_reader import VideoReader
from video_processing.frame_processor import FrameProcessor
from ai_engine.detection.yolo_detector import PersonDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Test YOLO person detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for annotated video"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display processed frames"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip (0 = process all)"
    )
    
    args = parser.parse_args()
    
    # Validate video path
    if not Path(args.source).exists():
        logger.error(f"Video file not found: {args.source}")
        return 1
    
    logger.info(f"Starting YOLO detection demo")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence: {args.confidence}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Frame skip: {args.frame_skip}")
    
    # Initialize detector
    detector_config = {
        "device": args.device,
        "confidence_threshold": args.confidence,
        "image_size": args.imgsz,
        "verbose": False
    }
    
    detector = PersonDetector(config=detector_config)
    
    # Load model
    logger.info("Loading YOLO model...")
    if not detector.load_model():
        logger.error("Failed to load model")
        return 1
    
    logger.info(f"Model loaded: {detector.get_model_info()}")
    
    # Initialize video reader
    logger.info("Opening video...")
    try:
        with VideoReader(args.source) as reader:
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
            
            # Initialize frame processor
            processor = FrameProcessor(frame_skip=args.frame_skip)
            
            # Setup video writer if output specified
            video_writer = None
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"annotated_{Path(args.source).stem}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = metadata['fps'] if metadata['fps'] else 30
                video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (metadata['width'], metadata['height'])
                )
                logger.info(f"Output video: {output_path}")
            
            # Performance tracking
            total_frames = 0
            processed_frames = 0
            total_detections = 0
            start_time = time.time()
            
            # Process frames
            logger.info("Processing frames...")
            
            for frame, frame_number, timestamp in reader.frames(skip_frames=args.frame_skip):
                total_frames += 1
                
                # Check max frames limit
                if args.max_frames and processed_frames >= args.max_frames:
                    logger.info(f"Reached max frames limit: {args.max_frames}")
                    break
                
                # Skip if processor says so
                if not processor.should_process_frame(frame_number):
                    continue
                
                # Detect people
                detections = detector.detect(frame)
                processed_frames += 1
                total_detections += len(detections)
                
                # Draw detections
                annotated_frame = detector.draw_detections(frame, detections)
                
                # Add overlay info
                info_text = f"Frame: {frame_number} | People: {len(detections)} | FPS: {detector.get_performance_stats()['avg_latency_ms']:.1f}ms"
                cv2.putText(
                    annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Write to video
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Display frame if enabled
                if args.display:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break
                
                # Log progress every 30 frames
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames, {total_detections} total detections")
            
            # Cleanup
            if video_writer:
                video_writer.release()
            
            if args.display:
                cv2.destroyAllWindows()
        
        # Calculate performance
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get detector performance stats
        detector_stats = detector.get_performance_stats()
        
        logger.info("=" * 60)
        logger.info("DETECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames read: {total_frames}")
        logger.info(f"Frames processed: {processed_frames}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average detections per frame: {total_detections / processed_frames if processed_frames > 0 else 0:.2f}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        if processing_time > 0:
            effective_fps = processed_frames / processing_time
            logger.info(f"Effective processing FPS: {effective_fps:.2f}")
        
        logger.info("=" * 60)
        logger.info("DETECTOR PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Inference count: {detector_stats['inference_count']}")
        logger.info(f"Average latency: {detector_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Median latency: {detector_stats['median_latency_ms']:.2f}ms")
        logger.info(f"Min latency: {detector_stats['min_latency_ms']:.2f}ms")
        logger.info(f"Max latency: {detector_stats['max_latency_ms']:.2f}ms")
        logger.info("=" * 60)
        
        # Save performance report
        report = {
            "model_name": detector.get_model_info()["model_name"],
            "device": detector.get_device(),
            "input_video": str(Path(args.source).absolute()),
            "input_resolution": f"{metadata['width']}x{metadata['height']}",
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / processed_frames if processed_frames > 0 else 0,
            "avg_inference_latency_ms": detector_stats['avg_latency_ms'],
            "median_inference_latency_ms": detector_stats['median_latency_ms'],
            "min_inference_latency_ms": detector_stats['min_latency_ms'],
            "max_inference_latency_ms": detector_stats['max_latency_ms'],
            "processing_fps": processed_frames / processing_time if processing_time > 0 else 0,
            "confidence_threshold": args.confidence,
            "image_size": args.imgsz,
        }
        
        output_dir = Path(args.output) if args.output else Path("output/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "yolo_detection_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
