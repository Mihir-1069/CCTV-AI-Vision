"""
Demo script for testing Pose Estimation.
Accepts a video path and performs YOLO11n-pose inference.

Example usage:
    python scripts/test_pose.py --source data/videos/test_sample.mp4
    python scripts/test_pose.py --source data/videos/your_video.mp4 --output output/annotated_videos
    python scripts/test_pose.py --source data/videos/your_video.mp4 --display --max-frames 100
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
from ai_engine.pose.pose_estimator import PoseEstimator
from ai_engine.pose.pose_analyzer import PoseAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Test YOLO11n-pose estimation"
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
        help="Confidence threshold for person detection"
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for keypoint visibility"
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

    logger.info(f"Starting Pose Estimation demo")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence: {args.confidence}")
    logger.info(f"Keypoint threshold: {args.keypoint_threshold}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Frame skip: {args.frame_skip}")

    # Initialize pose estimator
    estimator_config = {
        "device": args.device,
        "confidence_threshold": args.confidence,
        "keypoint_threshold": args.keypoint_threshold,
        "image_size": args.imgsz,
        "verbose": False,
    }

    estimator = PoseEstimator(config=estimator_config)

    # Load model
    logger.info("Loading YOLO11n-pose model...")
    model_start = time.time()
    if not estimator.load_model():
        logger.error("Failed to load pose model")
        return 1

    model_load_time = time.time() - model_start
    logger.info(f"Model loaded in {model_load_time:.2f}s")
    logger.info(f"Model info: {estimator.get_model_info()}")

    # Initialize pose analyzer
    analyzer = PoseAnalyzer(keypoint_threshold=args.keypoint_threshold)

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
                output_path = output_dir / f"pose_{Path(args.source).stem}.mp4"

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
            total_poses = 0
            total_keypoints_visible = 0
            start_time = time.time()

            # Pose results for JSON output
            all_pose_results = []

            # Process frames
            logger.info("Processing frames...")

            for frame, frame_number, timestamp in reader.frames(skip_frames=args.frame_skip):
                total_frames += 1

                # Check max frames limit
                if args.max_frames and processed_frames >= args.max_frames:
                    logger.info(f"Reached max frames limit: {args.max_frames}")
                    break

                if not processor.should_process_frame(frame_number):
                    continue

                # Estimate poses
                pose_start = time.time()
                poses = estimator.estimate(frame)
                pose_latency = (time.time() - pose_start) * 1000
                processed_frames += 1
                total_poses += len(poses)

                # Analyze each pose
                for pose in poses:
                    # Count visible keypoints
                    visible = int(np.sum(pose["keypoints_conf"] >= args.keypoint_threshold))
                    total_keypoints_visible += visible

                    # Geometric analysis
                    analysis = analyzer.analyze(
                        pose["keypoints_xy"],
                        pose["keypoints_conf"]
                    )

                    # Build JSON-serializable pose result
                    pose_json = {
                        "pose_confidence": pose["confidence"],
                        "bbox": pose["bbox"],
                        "keypoints": pose["keypoints"],
                        "analysis": analysis,
                    }
                    all_pose_results.append({
                        "frame_number": frame_number,
                        "pose": pose_json,
                    })

                # Draw poses
                annotated_frame = estimator.draw_poses(frame, poses)

                # Add overlay info
                info_text = (
                    f"Frame: {frame_number} | Poses: {len(poses)} | "
                    f"Latency: {pose_latency:.1f}ms"
                )
                cv2.putText(
                    annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                # Write to video
                if video_writer:
                    video_writer.write(annotated_frame)

                # Display frame if enabled
                if args.display:
                    cv2.imshow('Pose Estimation', annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break

                # Log progress
                if processed_frames % 30 == 0:
                    logger.info(
                        f"Processed {processed_frames} frames, "
                        f"{total_poses} total poses, "
                        f"latency: {pose_latency:.1f}ms"
                    )

            # Cleanup
            if video_writer:
                video_writer.release()

            if args.display:
                cv2.destroyAllWindows()

        # Calculate performance
        end_time = time.time()
        processing_time = end_time - start_time

        # Get estimator performance stats
        est_stats = estimator.get_performance_stats()

        logger.info("=" * 60)
        logger.info("POSE ESTIMATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames read: {total_frames}")
        logger.info(f"Frames processed: {processed_frames}")
        logger.info(f"Total poses detected: {total_poses}")
        logger.info(f"Total visible keypoints: {total_keypoints_visible}")
        logger.info(f"Model load time: {model_load_time:.2f}s")
        logger.info(f"Processing time: {processing_time:.2f}s")

        if processing_time > 0:
            effective_fps = processed_frames / processing_time
            logger.info(f"Effective processing FPS: {effective_fps:.2f}")

        logger.info("=" * 60)
        logger.info("POSE ESTIMATOR PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Inference count: {est_stats['inference_count']}")
        if est_stats['avg_latency_ms'] is not None:
            logger.info(f"Average latency: {est_stats['avg_latency_ms']:.2f}ms")
            logger.info(f"Median latency: {est_stats['median_latency_ms']:.2f}ms")
            logger.info(f"Min latency: {est_stats['min_latency_ms']:.2f}ms")
            logger.info(f"Max latency: {est_stats['max_latency_ms']:.2f}ms")
        logger.info("=" * 60)

        # Save performance report
        output_dir = Path(args.output) if args.output else Path("output/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "model_name": estimator.get_model_info()["model_name"],
            "device": estimator.get_device(),
            "input_video": str(Path(args.source).absolute()),
            "input_resolution": f"{metadata['width']}x{metadata['height']}",
            "processed_frames": processed_frames,
            "total_poses": total_poses,
            "total_visible_keypoints": total_keypoints_visible,
            "model_load_time_s": round(model_load_time, 2),
            "avg_pose_latency_ms": est_stats.get('avg_latency_ms'),
            "median_pose_latency_ms": est_stats.get('median_latency_ms'),
            "min_pose_latency_ms": est_stats.get('min_latency_ms'),
            "max_pose_latency_ms": est_stats.get('max_latency_ms'),
            "processing_fps": processed_frames / processing_time if processing_time > 0 else 0,
            "confidence_threshold": args.confidence,
            "keypoint_threshold": args.keypoint_threshold,
            "image_size": args.imgsz,
        }

        report_path = output_dir / "pose_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved: {report_path}")

        # Save pose results JSON
        if all_pose_results:
            results_path = output_dir / "pose_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_pose_results, f, indent=2)
            logger.info(f"Pose results saved: {results_path}")
        else:
            logger.info("No poses detected — results file not saved")

        return 0

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
