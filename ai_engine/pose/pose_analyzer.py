"""
Pose Analyzer Module
Provides PoseAnalyzer class for geometric pose analysis.

Calculates joint angles and body orientation from COCO 17 keypoints.
All calculations are purely geometric — no inference of emotions,
intent, medical conditions, or psychological states.

Limitations:
- Angles are 2D projections and may not reflect true 3D joint angles.
- Body orientation is a rough estimate based on visible keypoints.
- Results are unreliable when keypoint confidence is low.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """
    Geometric pose analysis from COCO 17 keypoints.

    All methods handle missing keypoints safely by returning None
    for any calculation that requires unavailable data.
    """

    # Index mapping for COCO keypoints
    # 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
    # 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
    # 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
    # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

    def __init__(self, keypoint_threshold: float = 0.5):
        """
        Initialize PoseAnalyzer.

        Args:
            keypoint_threshold: Minimum confidence for a keypoint to be
                considered reliable.
        """
        self.keypoint_threshold = keypoint_threshold
        logger.info(f"PoseAnalyzer initialized (threshold={keypoint_threshold})")

    def analyze(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform geometric analysis on a single person's keypoints.

        Args:
            keypoints_xy: Array of shape (17, 2) with pixel coordinates.
            keypoints_conf: Array of shape (17,) with confidence scores.

        Returns:
            Dictionary with joint angles, body orientation, and visibility stats.
            Any value that cannot be calculated due to missing keypoints is None.
        """
        kp = self._keypoints_dict(keypoints_xy, keypoints_conf)

        result: Dict[str, Any] = {
            "joint_angles": {
                "left_elbow": self._joint_angle(kp, 5, 7, 9),    # shoulder-elbow-wrist
                "right_elbow": self._joint_angle(kp, 6, 8, 10),
                "left_knee": self._joint_angle(kp, 11, 13, 15),  # hip-knee-ankle
                "right_knee": self._joint_angle(kp, 12, 14, 16),
                "left_hip": self._joint_angle(kp, 5, 11, 13),    # shoulder-hip-knee
                "right_hip": self._joint_angle(kp, 6, 12, 14),
                "left_shoulder": self._joint_angle(kp, 11, 5, 7),  # hip-shoulder-elbow
                "right_shoulder": self._joint_angle(kp, 12, 6, 8),
            },
            "body_orientation": self._body_orientation(kp),
            "visibility_stats": self._visibility_stats(keypoints_conf),
        }

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keypoints_dict(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray
    ) -> Dict[int, Optional[np.ndarray]]:
        """
        Build a dict mapping keypoint index to (x, y) or None if low confidence.
        """
        kp: Dict[int, Optional[np.ndarray]] = {}
        for i in range(len(keypoints_conf)):
            if float(keypoints_conf[i]) >= self.keypoint_threshold:
                kp[i] = np.array(keypoints_xy[i], dtype=np.float64)
            else:
                kp[i] = None
        return kp

    def _joint_angle(
        self,
        kp: Dict[int, Optional[np.ndarray]],
        a_idx: int,
        b_idx: int,
        c_idx: int
    ) -> Optional[float]:
        """
        Calculate the angle at joint b_idx between points a_idx and c_idx.

        Uses the law of cosines:
            angle = arccos((|ab|² + |bc|² - |ac|²) / (2 * |ab| * |bc|))

        Args:
            kp: Keypoint dict from _keypoints_dict.
            a_idx: Index of point A (one end of the angle).
            b_idx: Index of point B (vertex of the angle).
            c_idx: Index of point C (other end of the angle).

        Returns:
            Angle in degrees [0, 180], or None if any keypoint is unavailable.
        """
        a = kp.get(a_idx)
        b = kp.get(b_idx)
        c = kp.get(c_idx)

        if a is None or b is None or c is None:
            return None

        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)

        if ab <= 0 or bc <= 0:
            return None

        # Clamp for numerical safety
        cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)
        angle_deg = float(np.degrees(angle_rad))

        return round(angle_deg, 2)

    def _body_orientation(
        self,
        kp: Dict[int, Optional[np.ndarray]]
    ) -> str:
        """
        Estimate rough body orientation based on visible keypoints.

        This is a simple geometric estimate with clear limitations:
        - "front": both shoulders visible, nose between eyes, hips visible.
        - "left_side": left keypoints more visible than right, or
          left ear visible but right ear not.
        - "right_side": right keypoints more visible than left.
        - "back": shoulders visible but nose and face keypoints not visible.
        - "unknown": insufficient keypoints.

        Returns None if too few keypoints are available.
        """
        # Count visible keypoints by side
        left_kps = [1, 3, 5, 7, 9, 11, 13, 15]   # left eye, ear, shoulder, elbow, wrist, hip, knee, ankle
        right_kps = [2, 4, 6, 8, 10, 12, 14, 16]  # right equivalents

        left_count = sum(1 for i in left_kps if kp.get(i) is not None)
        right_count = sum(1 for i in right_kps if kp.get(i) is not None)

        nose = kp.get(0)
        left_shoulder = kp.get(5)
        right_shoulder = kp.get(6)
        left_ear = kp.get(3)
        right_ear = kp.get(4)

        # Need at least shoulders to estimate orientation
        if left_shoulder is None and right_shoulder is None:
            return "unknown"

        # Check if facing front (face keypoints visible)
        face_visible = nose is not None and (
            kp.get(1) is not None or kp.get(2) is not None
        )

        if face_visible and left_shoulder is not None and right_shoulder is not None:
            return "front"

        # Side detection: one ear visible but not the other
        if left_ear is not None and right_ear is None:
            return "left_side"
        if right_ear is not None and left_ear is None:
            return "right_side"

        # Asymmetric shoulder/hip visibility suggests side view
        if left_count > right_count and right_shoulder is None:
            return "left_side"
        if right_count > left_count and left_shoulder is None:
            return "right_side"

        # Shoulders visible but face not → likely back
        if left_shoulder is not None and right_shoulder is not None and not face_visible:
            return "back"

        return "unknown"

    def _visibility_stats(
        self,
        keypoints_conf: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute keypoint visibility and confidence statistics.
        """
        confs = np.array(keypoints_conf, dtype=np.float64)
        visible_count = int(np.sum(confs >= self.keypoint_threshold))
        total_count = len(confs)

        return {
            "visible_keypoints": visible_count,
            "total_keypoints": total_count,
            "visibility_ratio": round(visible_count / total_count, 4) if total_count > 0 else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 4) if total_count > 0 else 0.0,
            "max_confidence": round(float(np.max(confs)), 4) if total_count > 0 else 0.0,
            "min_confidence": round(float(np.min(confs)), 4) if total_count > 0 else 0.0,
        }
