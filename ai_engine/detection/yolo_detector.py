"""
YOLO Person Detector Module
Provides PersonDetector class for detecting people using YOLO11n.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
import yaml

from ultralytics import YOLO

from .detection_utils import DetectionUtils
from ..path_utils import get_config_path

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Person detector using Ultralytics YOLO11n.

    Detects people in images using pretrained YOLO11n model.
    Supports CPU inference and configurable parameters.
    """

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        """
        Initialize PersonDetector.

        Args:
            config: Configuration dictionary (overrides config file)
            config_path: Path to YAML configuration file
        """
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.model_name: str = "yolo11n.pt"
        self.confidence_threshold: float = 0.5
        self.iou_threshold: float = 0.45
        self.image_size: int = 640
        self.half_precision: bool = False
        self.target_classes: List[int] = [0]  # COCO person class
        self.max_detections: int = 100
        self.verbose: bool = True

        # Performance tracking
        self._inference_times: List[float] = []

        # Load configuration
        if config_path is None:
            config_path = get_config_path("model.yaml")

        if Path(config_path).exists():
            self._load_config(config_path, config)
        elif config:
            self._apply_config(config)
        else:
            logger.warning("No configuration provided, using defaults")

        # Auto-detect device if set to "auto"
        if self.device == "auto":
            self.device = self._detect_device()

    def _load_config(self, config_path: str, override_config: Optional[dict] = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
            override_config: Optional override configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Apply YOLO configuration
            yolo_config = config.get('yolo', {})
            self.model_name = yolo_config.get('model_name', self.model_name)
            self.confidence_threshold = yolo_config.get('confidence_threshold', self.confidence_threshold)
            self.iou_threshold = yolo_config.get('iou_threshold', self.iou_threshold)
            self.max_detections = yolo_config.get('max_detections', self.max_detections)
            self.target_classes = yolo_config.get('target_classes', self.target_classes)
            self.device = yolo_config.get('device', self.device)
            self.half_precision = yolo_config.get('half_precision', self.half_precision)
            self.verbose = yolo_config.get('verbose', self.verbose)

            # Apply performance configuration for image size
            perf_config = config.get('performance', {})
            self.image_size = perf_config.get('resize_width', self.image_size)

            # Apply overrides
            if override_config:
                self._apply_config(override_config)

            logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def _apply_config(self, config: dict):
        """
        Apply configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        if 'model_name' in config:
            self.model_name = config['model_name']
        if 'confidence_threshold' in config:
            self.confidence_threshold = config['confidence_threshold']
        if 'iou_threshold' in config:
            self.iou_threshold = config['iou_threshold']
        if 'image_size' in config:
            self.image_size = config['image_size']
        if 'device' in config:
            self.device = config['device']
        if 'half_precision' in config:
            self.half_precision = config['half_precision']
        if 'target_classes' in config:
            self.target_classes = config['target_classes']
        if 'max_detections' in config:
            self.max_detections = config['max_detections']
        if 'verbose' in config:
            self.verbose = config['verbose']

    def _detect_device(self) -> str:
        """
        Auto-detect available device.

        Returns:
            Device string ('cpu', 'cuda', or 'mps')
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
        Load YOLO model.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")

            # Resolve path relative to project root so it works regardless of CWD
            from ..path_utils import get_project_root as _root
            model_path = _root() / self.model_name
            resolved = str(model_path) if model_path.exists() else self.model_name

            self.model = YOLO(resolved)

            # Set device
            if self.device != "cpu":
                self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries with keys:
            - class_id: COCO class ID
            - class_name: Class name
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
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

        # Measure inference time
        start_time = time.time()

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
                classes=self.target_classes
            )

            # Record inference time
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)

            # Parse results
            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # Convert to int
                        bbox = [int(coord) for coord in box]

                        # Validate and clip bbox
                        height, width = frame.shape[:2]
                        bbox = DetectionUtils.clip_bbox(bbox, width, height)

                        # Get class name
                        class_name = DetectionUtils.get_class_name(cls_id)

                        detection = {
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "bbox": bbox,
                            "confidence": float(conf)
                        }

                        detections.append(detection)

            # Limit detections
            if len(detections) > self.max_detections:
                detections = detections[:self.max_detections]

            if self.verbose and len(detections) > 0:
                logger.debug(f"Detected {len(detections)} person(s)")

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()

        for det in detections:
            bbox = det.get('bbox', [])
            label = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)

            if len(bbox) == 4:
                frame_copy = DetectionUtils.draw_bbox(
                    frame_copy, bbox, label, confidence, color, thickness
                )

        return frame_copy

    def get_device(self) -> str:
        """
        Get current device.

        Returns:
            Device string
        """
        return self.device

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "half_precision": self.half_precision,
            "target_classes": self.target_classes,
            "max_detections": self.max_detections,
            "is_loaded": self.model is not None,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
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
            "avg_latency_ms": np.mean(times_ms),
            "median_latency_ms": np.median(times_ms),
            "min_latency_ms": np.min(times_ms),
            "max_latency_ms": np.max(times_ms),
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._inference_times = []

    def __repr__(self) -> str:
        """String representation."""
        loaded = "loaded" if self.model else "not loaded"
        return f"PersonDetector({self.model_name}, {self.device}, {loaded})"
