"""
Pose Estimation Module
Provides PoseEstimator class for human pose estimation using YOLO11n-pose.

Uses the pretrained YOLO11n-pose model from Ultralytics to extract:
- Person bounding boxes
- COCO 17 keypoints (x, y, confidence per keypoint)
- Pose confidence scores

All tracking is anonymous — no facial recognition or permanent identity.
CPU inference is fully supported.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
import yaml

from ultralytics import YOLO

from .pose_utils import PoseUtils

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Human pose estimator using Ultralytics YOLO11n-pose.

    Detects people and extracts COCO 17 keypoints per person.
    Supports CPU inference and configurable parameters.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize PoseEstimator.

        Args:
            config: Configuration dictionary (overrides config file).
            config_path: Path to YAML configuration file.
        """
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.model_name: str = "yolo11n-pose.pt"
        self.confidence_threshold: float = 0.5
        self.keypoint_threshold: float = 0.5
        self.image_size: int = 640
        self.iou_threshold: float = 0.45
        self.association_iou_threshold: float = 0.5
        self.verbose: bool = True

        # Performance tracking
        self._inference_times: List[float] = []

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "model.yaml"

        if Path(config_path).exists():
            self._load_config(config_path, config)
        elif config:
            self._apply_config(config)
        else:
            logger.warning("No configuration provided, using defaults")

        # Auto-detect device if set to "auto"
        if self.device == "auto":
            self.device = self._detect_device()

    def _load_config(
        self,
        config_path: str,
        override_config: Optional[dict] = None
    ):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.
            override_config: Optional override configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Apply pose configuration
            pose_config = config.get('pose', {})
            self.model_name = pose_config.get('model_name', self.model_name)
            self.confidence_threshold = pose_config.get('confidence_threshold', self.confidence_threshold)
            self.keypoint_threshold = pose_config.get('keypoint_threshold', self.keypoint_threshold)
            self.device = pose_config.get('device', self.device)
            self.verbose = pose_config.get('verbose', self.verbose)
            self.iou_threshold = pose_config.get('iou_threshold', self.iou_threshold)
            self.association_iou_threshold = pose_config.get('association_iou_threshold', self.association_iou_threshold)

            # Pose-specific image size (independent of performance.resize_width)
            self.image_size = pose_config.get('image_size', self.image_size)

            # Apply overrides
            if override_config:
                self._apply_config(override_config)

            logger.info(f"Pose configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def _apply_config(self, config: dict):
        """
        Apply configuration dictionary.

        Args:
            config: Configuration dictionary.
        """
        if 'model_name' in config:
            self.model_name = config['model_name']
        if 'confidence_threshold' in config:
            self.confidence_threshold = config['confidence_threshold']
        if 'keypoint_threshold' in config:
            self.keypoint_threshold = config['keypoint_threshold']
        if 'image_size' in config:
            self.image_size = config['image_size']
        if 'iou_threshold' in config:
            self.iou_threshold = config['iou_threshold']
        if 'association_iou_threshold' in config:
            self.association_iou_threshold = config['association_iou_threshold']
        if 'device' in config:
            self.device = config['device']
        if 'verbose' in config:
            self.verbose = config['verbose']

    def _detect_device(self) -> str:
        """
        Auto-detect available device.

        Returns:
            Device string ('cpu', 'cuda', or 'mps').
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA detected, using GPU")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("MPS detected, using Apple Silicon GPU")
                return "mps"
        except ImportError:
            pass

        logger.info("No GPU detected, using CPU")
        return "cpu"

    def load_model(self) -> bool:
        """
        Load the YOLO11n-pose model.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Loading pose model: {self.model_name}")

            self.model = YOLO(self.model_name)

            if self.device != "cpu":
                self.model.to(self.device)

            logger.info(f"Pose model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
            return False

    def estimate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate poses in a frame.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of pose result dictionaries, one per detected person:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "keypoints_xy": np.ndarray (17, 2),
                "keypoints_conf": np.ndarray (17,),
                "keypoints": dict  # structured by COCO name with x, y, confidence or None
            }
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("Invalid frame provided")
            return []

        if len(frame.shape) < 2:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return []

        start_time = time.time()

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False
            )

            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)

            poses: List[Dict[str, Any]] = []

            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                return []

            # Extract bounding boxes
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            # Extract keypoints
            if r.keypoints is not None:
                kpts_xy = r.keypoints.xy.cpu().numpy()      # (N, 17, 2)
                kpts_conf = r.keypoints.conf.cpu().numpy()   # (N, 17)
            else:
                kpts_xy = np.zeros((len(boxes), 17, 2))
                kpts_conf = np.zeros((len(boxes), 17))

            for i in range(len(boxes)):
                bbox = [int(coord) for coord in boxes[i]]

                # Clip bbox to frame boundaries
                height, width = frame.shape[:2]
                bbox = self._clip_bbox(bbox, width, height)

                kp_xy = kpts_xy[i] if i < len(kpts_xy) else np.zeros((17, 2))
                kp_conf = kpts_conf[i] if i < len(kpts_conf) else np.zeros(17)

                # Build structured keypoint dict
                keypoints_dict = PoseUtils.keypoints_to_dict(
                    kp_xy, kp_conf, self.keypoint_threshold
                )

                # Pose confidence = box confidence
                pose_conf = float(confidences[i])

                pose_result = {
                    "bbox": bbox,
                    "confidence": pose_conf,
                    "keypoints_xy": kp_xy,
                    "keypoints_conf": kp_conf,
                    "keypoints": keypoints_dict,
                }

                poses.append(pose_result)

            if self.verbose and len(poses) > 0:
                logger.debug(f"Estimated poses for {len(poses)} person(s)")

            return poses

        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return []

    def estimate_with_tracking(
        self,
        frame: np.ndarray,
        tracking_output: List[Dict[str, Any]],
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate poses and associate them with existing tracking IDs.

        Uses greedy one-to-one IoU matching between pose bboxes and tracking bboxes.

        Args:
            frame: Input frame (BGR format).
            tracking_output: Tracking output from ByteTrackManager.update().
            iou_threshold: Minimum IoU for a valid association.
                If None, uses self.association_iou_threshold from config.

        Returns:
            List of pose dicts with added "track_id" and "association_status":
            - "matched": pose assigned to a track with IoU >= threshold.
            - "unmatched": no track has sufficient IoU.
            - "uncertain": best candidate already assigned to another pose.
            Unmatched/uncertain poses have track_id = None.
        """
        if iou_threshold is None:
            iou_threshold = self.association_iou_threshold

        poses = self.estimate(frame)

        if not poses:
            return []

        annotated = PoseUtils.associate_poses_to_tracks(
            poses, tracking_output, iou_threshold
        )

        return annotated

    def draw_poses(
        self,
        frame: np.ndarray,
        poses: List[Dict[str, Any]],
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        keypoint_color: Tuple[int, int, int] = (0, 255, 255),
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        bbox_color: Tuple[int, int, int] = (0, 200, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        keypoint_radius: int = 4
    ) -> np.ndarray:
        """
        Draw pose results on a frame.

        Args:
            frame: Input frame (BGR).
            poses: List of pose result dicts (optionally with track_id).
            draw_keypoints: Whether to draw keypoint circles.
            draw_skeleton: Whether to draw skeleton connections.
            draw_bbox: Whether to draw bounding boxes.
            keypoint_color: BGR colour for keypoints.
            skeleton_color: BGR colour for skeleton lines.
            bbox_color: BGR colour for bounding boxes.
            text_color: BGR colour for track ID text.
            thickness: Line thickness for skeleton and bbox.
            keypoint_radius: Radius for keypoint circles.

        Returns:
            Frame with poses drawn (original is not modified).
        """
        frame_copy = frame.copy()

        for pose in poses:
            bbox = pose.get("bbox", [])
            kp_xy = pose.get("keypoints_xy")
            kp_conf = pose.get("keypoints_conf")
            track_id = pose.get("track_id")
            confidence = pose.get("confidence", 0.0)

            # Draw bounding box
            if draw_bbox and len(bbox) == 4:
                cv2.rectangle(
                    frame_copy,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    bbox_color,
                    thickness
                )

            # Draw skeleton
            if draw_skeleton and kp_xy is not None and kp_conf is not None:
                frame_copy = PoseUtils.draw_skeleton(
                    frame_copy, kp_xy, kp_conf,
                    threshold=self.keypoint_threshold,
                    color=skeleton_color,
                    thickness=thickness
                )

            # Draw keypoints
            if draw_keypoints and kp_xy is not None and kp_conf is not None:
                frame_copy = PoseUtils.draw_keypoints(
                    frame_copy, kp_xy, kp_conf,
                    threshold=self.keypoint_threshold,
                    color=keypoint_color,
                    radius=keypoint_radius
                )

            # Draw track ID if available and confidently associated
            if track_id is not None and len(bbox) == 4:
                label = f"ID: {track_id} ({confidence:.2f})"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    frame_copy,
                    (bbox[0], bbox[1] - text_h - baseline - 5),
                    (bbox[0] + text_w + 5, bbox[1]),
                    bbox_color,
                    -1
                )
                cv2.putText(
                    frame_copy,
                    label,
                    (bbox[0] + 5, bbox[1] - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    2
                )

        return frame_copy

    @staticmethod
    def _clip_bbox(bbox: List[int], width: int, height: int) -> List[int]:
        """
        Clip bounding box to image boundaries.

        Args:
            bbox: [x1, y1, x2, y2].
            width: Image width.
            height: Image height.

        Returns:
            Clipped bounding box.
        """
        x1 = max(0, min(bbox[0], width))
        y1 = max(0, min(bbox[1], height))
        x2 = max(0, min(bbox[2], width))
        y2 = max(0, min(bbox[3], height))
        return [x1, y1, x2, y2]

    def get_device(self) -> str:
        """
        Get current device.

        Returns:
            Device string.
        """
        return self.device

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "is_loaded": self.model is not None,
            "keypoint_count": 17,
            "keypoint_names": PoseUtils.COCO_KEYPOINT_NAMES,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics.
        """
        if not self._inference_times:
            return {
                "inference_count": 0,
                "avg_latency_ms": None,
                "median_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
            }

        times_ms = [t * 1000 for t in self._inference_times]

        return {
            "inference_count": len(self._inference_times),
            "avg_latency_ms": float(np.mean(times_ms)),
            "median_latency_ms": float(np.median(times_ms)),
            "min_latency_ms": float(np.min(times_ms)),
            "max_latency_ms": float(np.max(times_ms)),
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._inference_times = []

    def __repr__(self) -> str:
        """String representation."""
        loaded = "loaded" if self.model else "not loaded"
        return f"PoseEstimator({self.model_name}, {self.device}, {loaded})"
