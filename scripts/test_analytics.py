"""
Demo script for testing Occupancy Analytics.
Accepts a video path and performs detection, tracking, and analytics.
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
from ai_engine.tracking.bytetrack_manager import ByteTrackManager
from ai_engine.analytics.occupancy_manager import OccupancyManager
from ai_engine.analytics.entry_exit_counter import EntryExitCounter
from ai_engine.analytics.zone_manager import ZoneManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Test YOLO detection + ByteTrack tracking + Occupancy Analytics"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_001",
        help="Camera ID"
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
        help="Output directory for annotated video and JSON"
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
    
    logger.info(f"Starting YOLO + ByteTrack + Analytics demo")
    logger.info(f"Source: {args.source}")
    logger.info(f"Camera ID: {args.camera_id}")
    logger.info(f"Confidence: {args.confidence}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {args.device}")
    
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
    
    # Initialize tracker
    logger.info("Initializing ByteTrack tracker...")
    tracker = ByteTrackManager(config={"verbose": False})
    
    if tracker.tracker is None:
        logger.error("Failed to initialize tracker")
        return 1
    
    logger.info(f"Tracker initialized: {tracker}")
    
    # Initialize analytics
    logger.info("Initializing analytics modules...")
    occupancy_manager = OccupancyManager(camera_id=args.camera_id)
    
    # Entry/exit counter - loads from camera.yaml config
    entry_exit_counter = EntryExitCounter(camera_id=args.camera_id)
    
    # Zone manager - loads from camera.yaml config
    zone_manager = ZoneManager(camera_id=args.camera_id)
    
    logger.info("Analytics modules initialized")
    
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
                output_path = output_dir / f"analytics_{Path(args.source).stem}.mp4"
                
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
            
            # Analytics history
            analytics_history = []
            
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
                det_start = time.time()
                detections = detector.detect(frame)
                det_latency = (time.time() - det_start) * 1000
                processed_frames += 1
                total_detections += len(detections)
                
                # Update tracker
                track_start = time.time()
                tracking_output = tracker.update(detections, frame)
                track_latency = (time.time() - track_start) * 1000
                
                # Update analytics
                occupancy_stats = occupancy_manager.update(tracking_output, frame_number)
                entry_exit_counter.update(tracking_output, frame_number)
                zone_manager.update_zones(tracking_output)
                
                # Combine analytics output
                analytics_output = {
                    "camera_id": args.camera_id,
                    "timestamp": occupancy_stats["timestamp"],
                    "frame_number": frame_number,
                    "occupancy": occupancy_stats["occupancy"],
                    "entry_exit": entry_exit_counter.get_counts()["entry_exit"],
                    "zones": zone_manager.get_zones_output()
                }
                
                analytics_history.append(analytics_output)
                
                # Draw tracks
                annotated_frame = tracker.draw_tracks(frame, tracking_output)
                
                # Draw zones
                for zone_name, zone_data in zone_manager._zones.items():
                    polygon = zone_data["polygon"]
                    if len(polygon) >= 3:
                        pts = np.array(polygon, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)
                
                # Draw entry/exit line
                cv2.line(
                    annotated_frame,
                    entry_exit_counter.line_start,
                    entry_exit_counter.line_end,
                    (0, 0, 255),
                    2
                )
                
                # Add overlay info
                active_tracks = len(tracker.get_active_tracks())
                occupancy = occupancy_manager.get_current_count()
                entries = entry_exit_counter.get_total_entries()
                exits = entry_exit_counter.get_total_exits()
                
                info_lines = [
                    f"Frame: {frame_number}",
                    f"Occupancy: {occupancy}",
                    f"Entries: {entries}",
                    f"Exits: {exits}",
                    f"Net: {entries - exits}",
                    f"Det: {det_latency:.1f}ms",
                    f"Track: {track_latency:.1f}ms"
                ]
                
                y_offset = 30
                for line in info_lines:
                    cv2.putText(
                        annotated_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    y_offset += 25
                
                # Write to video
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Display frame if enabled
                if args.display:
                    cv2.imshow('Analytics Demo', annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break
                
                # Log progress every 30 frames
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames, occupancy: {occupancy}")
            
            # Cleanup
            if video_writer:
                video_writer.release()
            
            if args.display:
                cv2.destroyAllWindows()
        
        # Calculate performance
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get performance stats
        detector_stats = detector.get_performance_stats()
        tracker_stats = tracker.get_tracking_stats()
        
        logger.info("=" * 60)
        logger.info("ANALYTICS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames read: {total_frames}")
        logger.info(f"Frames processed: {processed_frames}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        if processing_time > 0:
            effective_fps = processed_frames / processing_time
            logger.info(f"Effective processing FPS: {effective_fps:.2f}")
        
        logger.info("=" * 60)
        logger.info("OCCUPANCY STATS")
        logger.info("=" * 60)
        final_stats = occupancy_manager.get_current_stats()
        logger.info(f"Current count: {final_stats['occupancy']['current_count']}")
        logger.info(f"Max count: {final_stats['occupancy']['max_count']}")
        logger.info(f"Min count: {final_stats['occupancy']['min_count']}")
        logger.info(f"Average count: {final_stats['occupancy']['average_count']:.2f}")
        
        logger.info("=" * 60)
        logger.info("ENTRY/EXIT STATS")
        logger.info("=" * 60)
        ee_counts = entry_exit_counter.get_counts()
        logger.info(f"Total entries: {ee_counts['entry_exit']['total_entries']}")
        logger.info(f"Total exits: {ee_counts['entry_exit']['total_exits']}")
        logger.info(f"Net count: {ee_counts['entry_exit']['net_count']}")
        
        logger.info("=" * 60)
        logger.info("ZONE STATS")
        logger.info("=" * 60)
        zones = zone_manager.get_all_zones()
        for zone_name, zone_data in zones.items():
            logger.info(f"{zone_name}: {zone_data['count']} people")
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Detection latency: {detector_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Tracking latency: {tracker_stats['avg_tracking_latency_ms']:.2f}ms")
        
        # Save analytics report
        output_dir = Path(args.output) if args.output else Path("output/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "camera_id": args.camera_id,
            "input_video": str(Path(args.source).absolute()),
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "occupancy": final_stats["occupancy"],
            "entry_exit": ee_counts["entry_exit"],
            "zones": zone_manager.get_zones_output(),
            "performance": {
                "avg_detection_latency_ms": detector_stats['avg_latency_ms'],
                "avg_tracking_latency_ms": tracker_stats['avg_tracking_latency_ms'],
                "processing_fps": processed_frames / processing_time if processing_time > 0 else 0
            }
        }
        
        report_path = output_dir / "analytics_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analytics report saved: {report_path}")
        
        # Save analytics history
        history_path = output_dir / "analytics_history.json"
        with open(history_path, 'w') as f:
            json.dump(analytics_history, f, indent=2)
        
        logger.info(f"Analytics history saved: {history_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
