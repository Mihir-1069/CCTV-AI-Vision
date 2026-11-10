"""
Inference Pipeline — SchoolVision AI
Reusable wrapper that assembles and runs the full CV pipeline on a single frame.

Motivation
----------
``scripts/run_video.py`` contains all pipeline initialisation and frame-loop
logic.  This module extracts the *per-frame* processing step into a reusable
class so that future integrations (FastAPI endpoints, notebook demos, unit
tests) can call ``pipeline.process_frame(frame, frame_number)`` without
duplicating the orchestration logic.

``run_video.py`` remains the primary CLI entry point and is NOT replaced.

Design rules
------------
- All component classes are imported from their existing modules unchanged.
- Component initialisation is deferred to ``InferencePipeline.build()`` so that
  the pipeline can be constructed lazily.
- Optional components (anomaly, event engine, crowd analytics) are degraded
  gracefully when their modules are not yet fully implemented.
- No new behaviour is added beyond what already exists in ``run_video.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ai_engine.detection.yolo_detector import PersonDetector
from ai_engine.tracking.bytetrack_manager import ByteTrackManager
from ai_engine.pose.pose_estimator import PoseEstimator
from ai_engine.activity.activity_recognition import ActivityRecognizer
from ai_engine.analytics.occupancy_manager import OccupancyManager

# Optional components — graceful degradation
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

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """
    Per-frame output from InferencePipeline.process_frame().

    All fields mirror the data structures already produced by the individual
    pipeline components.  No new schemas are introduced.
    """
    frame_number:   int
    camera_id:      str
    tracking_output: List[Dict[str, Any]] = field(default_factory=list)
    occupancy_stats: Dict[str, Any]       = field(default_factory=dict)
    pose_results:    List[Dict[str, Any]] = field(default_factory=list)
    enriched_poses:  List[Dict[str, Any]] = field(default_factory=list)
    anomaly_results: List[Dict[str, Any]] = field(default_factory=list)
    alerts:          List[Any]            = field(default_factory=list)
    crowd_stats:     Optional[Dict[str, Any]] = None


class InferencePipeline:
    """
    Full-frame inference pipeline for SchoolVision AI.

    Usage
    -----
    ::

        pipeline = InferencePipeline(camera_id="cam_001")
        pipeline.build()

        for frame, frame_number, _ in reader.frames():
            result = pipeline.process_frame(frame, frame_number)
            annotated = pipeline.annotate(frame, result)
    """

    def __init__(
        self,
        camera_id: str = "cam_001",
        config_path: Optional[str] = None,
        frame_width: int = 640,
        fps: float = 30.0,
        enable_pose: bool = True,
        enable_activity: bool = True,
        enable_anomaly: bool = True,
        enable_events: bool = True,
        enable_crowd: bool = True,
    ) -> None:
        self.camera_id    = camera_id
        self.config_path  = config_path
        self.frame_width  = frame_width
        self.fps          = fps

        self._enable_pose     = enable_pose
        self._enable_activity = enable_activity
        self._enable_anomaly  = enable_anomaly and _HAS_ANOMALY
        self._enable_events   = enable_events and _HAS_EVENTS
        self._enable_crowd    = enable_crowd and _HAS_CROWD

        # Component references — populated in build()
        self.detector:        Optional[PersonDetector]    = None
        self.tracker:         Optional[ByteTrackManager]  = None
        self.occupancy:       Optional[OccupancyManager]  = None
        self.pose_estimator:  Optional[PoseEstimator]     = None
        self.recognizer:      Optional[ActivityRecognizer] = None
        self.anomaly_detector: Optional[Any]              = None
        self.rule_engine:     Optional[Any]               = None
        self.alert_gen:       Optional[Any]               = None
        self.correlator:      Optional[Any]               = None
        self.crowd_analyzer:  Optional[Any]               = None

        self._built = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def build(self) -> "InferencePipeline":
        """
        Instantiate and load all pipeline components.

        Raises RuntimeError if the detection model cannot be loaded (required).
        All other components fail gracefully with a warning.
        """
        logger.info("Building InferencePipeline for camera: %s", self.camera_id)

        self.detector = PersonDetector(config_path=self.config_path)
        if not self.detector.load_model():
            raise RuntimeError("Detection model failed to load — cannot build pipeline")

        self.tracker  = ByteTrackManager(config_path=self.config_path)
        self.occupancy = OccupancyManager(camera_id=self.camera_id)

        if self._enable_pose:
            self.pose_estimator = PoseEstimator(config_path=self.config_path)
            if not self.pose_estimator.load_model():
                logger.warning("Pose model could not be loaded — disabling pose estimation")
                self.pose_estimator = None
                self._enable_pose     = False
                self._enable_activity = False

        if self._enable_activity and self.pose_estimator:
            self.recognizer = ActivityRecognizer(config_path=self.config_path)

        if self._enable_anomaly:
            try:
                self.anomaly_detector = AnomalyDetector()
                self.anomaly_detector.load_model()
            except Exception as exc:
                logger.warning("AnomalyDetector unavailable: %s", exc)
                self.anomaly_detector = None

        if self._enable_events:
            try:
                self.rule_engine = RuleEngine()
                self.alert_gen   = AlertGenerator()
                self.correlator  = EventCorrelator()
            except Exception as exc:
                logger.warning("Event engine unavailable: %s", exc)
                self.rule_engine = self.alert_gen = self.correlator = None

        if self._enable_crowd:
            try:
                self.crowd_analyzer = CrowdAnalytics()
            except Exception as exc:
                logger.warning("CrowdAnalytics unavailable: %s", exc)
                self.crowd_analyzer = None

        self._built = True
        logger.info("InferencePipeline built successfully")
        return self

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> FrameResult:
        """
        Run the full pipeline on one preprocessed frame.

        Args:
            frame:        BGR numpy array (already preprocessed by FrameProcessor).
            frame_number: Sequential frame index from the video reader.

        Returns:
            FrameResult containing all per-frame pipeline outputs.
        """
        if not self._built:
            raise RuntimeError("Call build() before process_frame()")

        result = FrameResult(frame_number=frame_number, camera_id=self.camera_id)

        # --- Detect ---
        detections = self.detector.detect(frame)

        # --- Track ---
        result.tracking_output = self.tracker.update(detections, frame)

        # --- Occupancy ---
        result.occupancy_stats = self.occupancy.update(result.tracking_output, frame_number)

        # --- Pose + Activity ---
        if self.pose_estimator and result.tracking_output:
            result.pose_results = self.pose_estimator.estimate_with_tracking(
                frame, result.tracking_output
            )

            if self.recognizer:
                result.enriched_poses = self.recognizer.update(
                    result.pose_results, frame_number,
                    frame_width=self.frame_width, fps=self.fps
                )
            else:
                result.enriched_poses = result.pose_results

            # --- Anomaly Detection ---
            if self.anomaly_detector:
                for pose in result.enriched_poses:
                    kp_xy   = pose.get("keypoints_xy")
                    kp_conf = pose.get("keypoints_conf")
                    if kp_xy is not None and kp_conf is not None:
                        score = self.anomaly_detector.score(kp_xy, kp_conf)
                        result.anomaly_results.append({
                            "track_id":     pose.get("track_id"),
                            "anomaly_score": score,
                            "is_anomaly":   self.anomaly_detector.is_anomaly(score),
                        })

        # --- Rule Engine + Alerts ---
        if self.rule_engine and self.alert_gen:
            frame_data = {
                "frame_number":    frame_number,
                "camera_id":       self.camera_id,
                "tracking_output": result.tracking_output,
                "occupancy_stats": result.occupancy_stats,
                "pose_results":    result.enriched_poses,
                "anomaly_results": result.anomaly_results,
            }
            events = self.rule_engine.evaluate(frame_data)
            for event in events:
                alert = self.alert_gen.generate(event)
                if alert:
                    result.alerts.append(alert)
            if self.correlator and result.alerts:
                for alert in result.alerts:
                    self.correlator.correlate(alert)

        # --- Crowd Analysis ---
        if self.crowd_analyzer and result.tracking_output:
            result.crowd_stats = self.crowd_analyzer.update(
                result.tracking_output,
                frame_number=frame_number,
                timestamp=None,
            )

        return result

    # ------------------------------------------------------------------
    # Annotation helper (mirrors run_video.py annotation block)
    # ------------------------------------------------------------------

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Draw tracking boxes, skeletons, activity labels and HUD onto a frame copy.

        Args:
            frame:  Original BGR frame (not modified).
            result: FrameResult from process_frame().

        Returns:
            Annotated copy of the frame.
        """
        import cv2  # local import to avoid top-level requirement

        annotated = frame.copy()
        annotated = self.tracker.draw_tracks(annotated, result.tracking_output)

        if self.pose_estimator and result.enriched_poses:
            annotated = self.pose_estimator.draw_poses(annotated, result.enriched_poses)

        for pose in result.enriched_poses:
            bbox = pose.get("bbox", [])
            act  = pose.get("activity", {})
            if len(bbox) == 4 and act.get("label"):
                label  = act["label"]
                colour = (0, 255, 0)   if label == "STANDING"       else \
                         (255, 165, 0) if label == "WALKING"         else \
                         (0, 0, 255)   if label == "FALL_LIKE_EVENT" else \
                         (200, 200, 200)
                cv2.putText(annotated, label,
                            (int(bbox[0]), int(bbox[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        occ_text = f"Count: {result.occupancy_stats.get('occupancy', {}).get('current_count', 0)}"
        cv2.putText(annotated, occ_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if result.alerts:
            cv2.putText(annotated, f"Alerts: {len(result.alerts)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated
