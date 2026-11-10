"""
Video Pipeline Runner — SchoolVision AI
Runs the full CV pipeline on a video file or RTSP stream.

Pipeline: VideoReader → FrameProcessor → PersonDetector → ByteTrackManager
          → PoseEstimator → ActivityRecognizer → OccupancyManager
          → AnomalyDetector → RuleEngine → AlertGenerator → EventCorrelator

Usage
-----
  python scripts/run_video.py --source data/videos/test_sample.mp4
  python scripts/run_video.py --source rtsp://camera_ip/stream --camera-id cam_001
  python scripts/run_video.py --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from ai_engine.detection.yolo_detector import PersonDetector
from ai_engine.tracking.bytetrack_manager import ByteTrackManager
from ai_engine.pose.pose_estimator import PoseEstimator
from ai_engine.activity.activity_recognition import ActivityRecognizer, ActivityLabel
from ai_engine.analytics.occupancy_manager import OccupancyManager
from video_processing.video_reader import VideoReader
from video_processing.frame_processor import FrameProcessor

# Optional imports — gracefully degraded when not yet implemented
try:
    from ai_engine.anomaly.autoencoder import AnomalyDetector
    _HAS_ANOMALY = True
except (ImportError, Exception):
    _HAS_ANOMALY = False

try:
    from event_engine.rule_engine import RuleEngine
    from event_engine.alert_generator import AlertGenerator
    from event_engine.event_correlator import EventCorrelator
    _HAS_EVENTS = True
except (ImportError, Exception):
    _HAS_EVENTS = False

try:
    from ai_engine.crowd.crowd_analytics import CrowdAnalytics
    _HAS_CROWD = True
except (ImportError, Exception):
    _HAS_CROWD = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_video")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SchoolVision AI — Video Pipeline Runner")
    p.add_argument("--source",     required=True,  help="Video file path or RTSP URL")
    p.add_argument("--camera-id",  default="cam_001", help="Camera identifier")
    p.add_argument("--config",     default=None,   help="Path to model.yaml (optional)")
    p.add_argument("--show",       action="store_true", help="Show live preview window")
    p.add_argument("--save",       default=None,   help="Output annotated video path")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    p.add_argument("--no-pose",    action="store_true", help="Skip pose estimation")
    p.add_argument("--no-activity",action="store_true", help="Skip activity recognition")
    return p


def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("=== SchoolVision AI Pipeline Starting ===")
    logger.info("Source: %s | Camera: %s", args.source, args.camera_id)

    # --- Initialise modules ---
    detector   = PersonDetector()
    tracker    = ByteTrackManager()
    occupancy  = OccupancyManager(camera_id=args.camera_id)
    recognizer = ActivityRecognizer(config_path=args.config) if not args.no_activity else None

    pose_estimator = None
    if not args.no_pose:
        pose_estimator = PoseEstimator(config_path=args.config)
        if not pose_estimator.load_model():
            logger.warning("Pose model could not be loaded — skipping pose estimation")
            pose_estimator = None

    if not detector.load_model():
        logger.error("Detection model failed to load — aborting")
        sys.exit(1)

    anomaly_detector = rule_engine = alert_gen = correlator = crowd_analyzer = None

    if _HAS_ANOMALY:
        try:
            anomaly_detector = AnomalyDetector()
            anomaly_detector.load_model()
            logger.info("AnomalyDetector loaded")
        except Exception as e:
            logger.warning("AnomalyDetector unavailable: %s", e)

    if _HAS_EVENTS:
        try:
            rule_engine  = RuleEngine()
            alert_gen    = AlertGenerator()
            correlator   = EventCorrelator()
            logger.info("Event engine loaded")
        except Exception as e:
            logger.warning("Event engine unavailable: %s", e)

    if _HAS_CROWD:
        try:
            crowd_analyzer = CrowdAnalyzer()
            logger.info("CrowdAnalyzer loaded")
        except Exception as e:
            logger.warning("CrowdAnalyzer unavailable: %s", e)

    # --- Open video source ---
    reader    = VideoReader(args.source)
    processor = FrameProcessor()

    if not reader.open():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    fps         = reader.fps or 30.0
    frame_width = reader.width or 640
    logger.info("Stream: %dx%d @ %.1f fps", frame_width, reader.height or 480, fps)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (frame_width, reader.height or 480))

    frame_number = 0
    t0 = time.time()

    while True:
        raw_frame, _, _ = reader.read_frame()
        if raw_frame is None:
            break
        if args.max_frames and frame_number >= args.max_frames:
            break

        frame = processor.preprocess_frame(raw_frame)
        if frame is None:
            frame_number += 1
            continue

        # --- Detect ---
        detections = detector.detect(frame)

        # --- Track ---
        tracking_output = tracker.update(detections, frame)

        # --- Occupancy ---
        occ_stats = occupancy.update(tracking_output, frame_number)

        # --- Pose + Activity ---
        pose_results      = []
        enriched_poses    = []
        anomaly_results   = []

        if pose_estimator and tracking_output:
            pose_results = pose_estimator.estimate_with_tracking(frame, tracking_output)

            if recognizer:
                enriched_poses = recognizer.update(
                    pose_results, frame_number,
                    frame_width=frame_width, fps=fps
                )
            else:
                enriched_poses = pose_results

            # --- Anomaly Detection ---
            if anomaly_detector:
                for pose in enriched_poses:
                    kp_xy   = pose.get("keypoints_xy")
                    kp_conf = pose.get("keypoints_conf")
                    if kp_xy is not None and kp_conf is not None:
                        score = anomaly_detector.score(kp_xy, kp_conf)
                        anomaly_results.append({
                            "track_id": pose.get("track_id"),
                            "anomaly_score": score,
                            "is_anomaly": anomaly_detector.is_anomaly(score),
                        })

        # --- Rule Engine + Alerts ---
        alerts = []
        if rule_engine and alert_gen:
            frame_data = {
                "frame_number":   frame_number,
                "camera_id":      args.camera_id,
                "tracking_output":tracking_output,
                "occupancy_stats":occ_stats,
                "pose_results":   enriched_poses,
                "anomaly_results":anomaly_results,
            }
            events = rule_engine.evaluate(frame_data)
            for event in events:
                alert = alert_gen.generate(event)
                if alert:
                    alerts.append(alert)
            if correlator and alerts:
                for alert in alerts:
                    correlator.correlate(alert)

        # --- Crowd Analysis ---
        crowd_stats = None
        if crowd_analyzer and tracking_output:
            crowd_stats = crowd_analyzer.update(
                tracking_output,
                frame_number=frame_number,
                timestamp=None,
            )

        # --- Annotate frame ---
        annotated = frame.copy()
        annotated = tracker.draw_tracks(annotated, tracking_output)
        if pose_estimator and enriched_poses:
            annotated = pose_estimator.draw_poses(annotated, enriched_poses)

        # Overlay activity labels
        for pose in enriched_poses:
            bbox = pose.get("bbox", [])
            act  = pose.get("activity", {})
            if len(bbox) == 4 and act.get("label"):
                label = act["label"]
                colour = (0, 255, 0) if label == "STANDING" else \
                         (255, 165, 0) if label == "WALKING"  else \
                         (0, 0, 255)   if label == "FALL_LIKE_EVENT" else \
                         (200, 200, 200)
                cv2.putText(annotated, label,
                            (int(bbox[0]), int(bbox[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Overlay occupancy
        occ_text = f"Count: {occ_stats['occupancy']['current_count']}"
        cv2.putText(annotated, occ_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Overlay alert count
        if alerts:
            cv2.putText(annotated, f"Alerts: {len(alerts)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if args.show:
            cv2.imshow("SchoolVision AI", annotated)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(annotated)

        frame_number += 1
        if frame_number % 100 == 0:
            elapsed = time.time() - t0
            logger.info("Frame %d | %.1f fps | People: %d",
                        frame_number, frame_number / max(elapsed, 0.001),
                        occ_stats["occupancy"]["current_count"])

    # --- Cleanup ---
    reader.close()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    logger.info("=== Pipeline Complete: %d frames in %.1fs (%.1f fps avg) ===",
                frame_number, elapsed, frame_number / max(elapsed, 0.001))


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_pipeline(args)
