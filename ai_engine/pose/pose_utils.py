"""
Pose Utilities Module
Provides utility functions for pose estimation operations.

Includes:
- COCO 17 keypoint names and skeleton connections
- Keypoint filtering and structured conversion
- Greedy one-to-one IoU association of poses to tracking IDs
- Drawing helpers for keypoints and skeleton
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PoseUtils:
    """
    Utility functions for pose estimation operations.
    """

    # COCO 17 keypoints — order must match YOLO11n-pose model output
    COCO_KEYPOINT_NAMES: List[str] = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # Skeleton connections as (index_a, index_b) pairs
    # Based on the standard COCO skeleton used in pose estimation
    SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1),   # nose - left_eye
        (0, 2),   # nose - right_eye
        (1, 3),   # left_eye - left_ear
        (2, 4),   # right_eye - right_ear
        (5, 6),   # left_shoulder - right_shoulder
        (5, 7),   # left_shoulder - left_elbow
        (7, 9),   # left_elbow - left_wrist
        (6, 8),   # right_shoulder - right_elbow
        (8, 10),  # right_elbow - right_wrist
        (5, 11),  # left_shoulder - left_hip
        (6, 12),  # right_shoulder - right_hip
        (11, 12), # left_hip - right_hip
        (11, 13), # left_hip - left_knee
        (13, 15), # left_knee - left_ankle
        (12, 14), # right_hip - right_knee
        (14, 16), # right_knee - right_ankle
    ]

    # ------------------------------------------------------------------
    # Keypoint conversion
    # ------------------------------------------------------------------

    @staticmethod
    def keypoints_to_dict(
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Convert raw keypoint arrays to a structured dictionary.

        Each keypoint becomes {"x": float, "y": float, "confidence": float}.
        Keypoints below the threshold are set to None.

        Args:
            keypoints_xy: Array of shape (17, 2) with x, y pixel coordinates.
            keypoints_conf: Array of shape (17,) with confidence scores.
            threshold: Minimum confidence to include a keypoint.

        Returns:
            Dictionary keyed by COCO name. Unavailable keypoints are None.
        """
        result: Dict[str, Optional[Dict[str, float]]] = {}

        for i, name in enumerate(PoseUtils.COCO_KEYPOINT_NAMES):
            conf = float(keypoints_conf[i]) if i < len(keypoints_conf) else 0.0

            if conf >= threshold:
                xy = keypoints_xy[i] if i < len(keypoints_xy) else np.array([0.0, 0.0])
                result[name] = {
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "confidence": conf,
                }
            else:
                result[name] = None

        return result

    @staticmethod
    def filter_keypoints(
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter keypoints by confidence threshold.

        Args:
            keypoints_xy: Array of shape (17, 2).
            keypoints_conf: Array of shape (17,).
            threshold: Minimum confidence.

        Returns:
            List of keypoint dicts with name, x, y, confidence for visible points.
        """
        filtered = []
        for i, name in enumerate(PoseUtils.COCO_KEYPOINT_NAMES):
            conf = float(keypoints_conf[i]) if i < len(keypoints_conf) else 0.0
            if conf >= threshold:
                xy = keypoints_xy[i] if i < len(keypoints_xy) else np.array([0.0, 0.0])
                filtered.append({
                    "name": name,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "confidence": conf,
                })
        return filtered

    # ------------------------------------------------------------------
    # IoU and tracking association
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU value in [0, 1].
        """
        x1_i = max(bbox1[0], bbox2[0])
        y1_i = max(bbox1[1], bbox2[1])
        x2_i = min(bbox1[2], bbox2[2])
        y2_i = min(bbox1[3], bbox2[3])

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def associate_poses_to_tracks(
        pose_results: List[Dict[str, Any]],
        tracking_output: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Associate pose detections with tracking track IDs using greedy IoU matching.

        Guarantees one-to-one assignment:
        - Each pose gets at most one track_id.
        - Each track_id is assigned to at most one pose per frame.

        Association statuses:
        - "matched": pose successfully assigned to a track with IoU >= threshold.
        - "unmatched": no track has IoU >= threshold for this pose.
        - "uncertain": best candidate track was already assigned to another pose
          (conflict resolved in favour of the higher-IoU pose).

        Args:
            pose_results: List of pose dicts with "bbox" key.
            tracking_output: List of tracking dicts with "bbox" and "track_id".
            iou_threshold: Minimum IoU for a valid association.

        Returns:
            List of pose dicts with added "track_id" and "association_status" keys.
            Unmatched/uncertain poses have track_id = None.
        """
        n_poses = len(pose_results)
        n_tracks = len(tracking_output)

        # Initialise results — copy pose dicts, add association fields
        annotated = []
        for p in pose_results:
            annotated.append(dict(p))
            annotated[-1]["track_id"] = None
            annotated[-1]["association_status"] = "unmatched"

        if n_poses == 0 or n_tracks == 0:
            return annotated

        # Build all (pose_idx, track_idx, iou) candidates
        candidates: List[Tuple[int, int, float]] = []
        for pi in range(n_poses):
            pose_bbox = annotated[pi].get("bbox", [0, 0, 0, 0])
            for ti in range(n_tracks):
                track_bbox = tracking_output[ti].get("bbox", [0, 0, 0, 0])
                iou = PoseUtils.calculate_iou(pose_bbox, track_bbox)
                if iou > 0:
                    candidates.append((pi, ti, iou))

        # Sort by IoU descending — greedy assignment
        candidates.sort(key=lambda x: x[2], reverse=True)

        assigned_poses: set = set()
        assigned_tracks: set = set()

        for pi, ti, iou in candidates:
            if pi in assigned_poses or ti in assigned_tracks:
                # This pose or track already assigned — check if uncertain
                if pi not in assigned_poses and iou >= iou_threshold:
                    # Pose was not assigned but its best candidate is taken
                    annotated[pi]["association_status"] = "uncertain"
                continue

            if iou >= iou_threshold:
                track_id = tracking_output[ti].get("track_id", -1)
                if track_id is not None and track_id >= 0:
                    annotated[pi]["track_id"] = track_id
                    annotated[pi]["association_status"] = "matched"
                    assigned_poses.add(pi)
                    assigned_tracks.add(ti)

        return annotated

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def draw_keypoints(
        frame: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        threshold: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 255),
        radius: int = 4
    ) -> np.ndarray:
        """
        Draw visible keypoints on the frame.

        Args:
            frame: Input frame (BGR).
            keypoints_xy: Array of shape (17, 2).
            keypoints_conf: Array of shape (17,).
            threshold: Minimum confidence to draw.
            color: BGR colour tuple.
            radius: Circle radius.

        Returns:
            Frame with keypoints drawn (original is not modified).
        """
        frame_copy = frame.copy()

        for i in range(len(keypoints_conf)):
            conf = float(keypoints_conf[i])
            if conf < threshold:
                continue

            x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
            cv2.circle(frame_copy, (x, y), radius, color, -1)

        return frame_copy

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        threshold: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw skeleton connections between visible keypoints.

        Only draws a connection if both endpoints are above the confidence threshold.

        Args:
            frame: Input frame (BGR).
            keypoints_xy: Array of shape (17, 2).
            keypoints_conf: Array of shape (17,).
            threshold: Minimum confidence for both endpoints.
            color: BGR colour tuple.
            thickness: Line thickness.

        Returns:
            Frame with skeleton drawn (original is not modified).
        """
        frame_copy = frame.copy()

        for idx_a, idx_b in PoseUtils.SKELETON_CONNECTIONS:
            if idx_a >= len(keypoints_conf) or idx_b >= len(keypoints_conf):
                continue

            if float(keypoints_conf[idx_a]) < threshold:
                continue
            if float(keypoints_conf[idx_b]) < threshold:
                continue

            xa, ya = int(keypoints_xy[idx_a][0]), int(keypoints_xy[idx_a][1])
            xb, yb = int(keypoints_xy[idx_b][0]), int(keypoints_xy[idx_b][1])

            cv2.line(frame_copy, (xa, ya), (xb, yb), color, thickness)

        return frame_copy
