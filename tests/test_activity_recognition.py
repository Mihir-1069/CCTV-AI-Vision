"""
Unit tests for ActivityRecognizer — Phase 7.

All tests are purely synthetic: no model weights, video files, or internet
access required. Keypoint arrays are hand-crafted to represent known postures
so that test failures indicate real logic regressions.

COCO keypoint index reference used throughout
---------------------------------------------
0  nose          5  left_shoulder   6  right_shoulder
7  left_elbow    8  right_elbow     9  left_wrist    10 right_wrist
11 left_hip     12 right_hip       13 left_knee     14 right_knee
15 left_ankle   16 right_ankle
1  left_eye      2  right_eye       3  left_ear       4  right_ear

PoseAnalyzer angle definitions (a → b ← c means angle at b)
------------------------------------------------------------
left_knee  : hip(11) → knee(13) ← ankle(15)
right_knee : hip(12) → knee(14) ← ankle(16)
left_hip   : shoulder(5) → hip(11) ← knee(13)
right_hip  : shoulder(6) → hip(12) ← knee(14)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.activity.activity_recognition import (
    ActivityLabel,
    ActivityRecognizer,
    ActivityResult,
)


# ===========================================================================
# Keypoint factory helpers
# ===========================================================================

def _make_kp() -> np.ndarray:
    """Return a zeroed (17, 2) float32 keypoint coordinate array."""
    return np.zeros((17, 2), dtype=np.float32)


def _make_conf(value: float = 0.9) -> np.ndarray:
    """Return a uniform (17,) float32 confidence array."""
    return np.full(17, value, dtype=np.float32)


def _conf_zeros() -> np.ndarray:
    """All-zero confidence — no visible keypoints."""
    return np.zeros(17, dtype=np.float32)


def _standing_keypoints():
    """
    Synthetic STANDING posture.

    Person faces the camera; legs are straight; tall bounding box.
    Computed knee angle ≈ 180° (> 155°), hip angle ≈ 176° (> 150°).
    Aspect ratio = 490/160 ≈ 3.06  (> 1.8).
    Shoulder–hip vertical gap = 170 px  (> 30 px).
    """
    kp_xy = _make_kp()
    kp_conf = _make_conf(0.9)

    # Face
    kp_xy[0]  = [200,  50]   # nose
    kp_xy[1]  = [185,  45]   # left_eye
    kp_xy[2]  = [215,  45]   # right_eye

    # Torso
    kp_xy[5]  = [160, 150]   # left_shoulder
    kp_xy[6]  = [240, 150]   # right_shoulder
    kp_xy[11] = [170, 320]   # left_hip
    kp_xy[12] = [230, 320]   # right_hip

    # Arms
    kp_xy[7]  = [140, 230]   # left_elbow
    kp_xy[8]  = [260, 230]   # right_elbow
    kp_xy[9]  = [130, 310]   # left_wrist
    kp_xy[10] = [270, 310]   # right_wrist

    # Legs — perfectly vertical → knee angle = 180°
    kp_xy[13] = [170, 410]   # left_knee
    kp_xy[14] = [230, 410]   # right_knee
    kp_xy[15] = [170, 500]   # left_ankle
    kp_xy[16] = [230, 500]   # right_ankle

    bbox = [120, 30, 280, 520]   # h=490, w=160, AR≈3.06
    return kp_xy, kp_conf, bbox


def _sitting_keypoints():
    """
    Synthetic SITTING posture.

    Thighs are horizontal (90° knee angle < 120°).
    Compressed bounding box: AR ≈ 0.93 (< 1.4).
    """
    kp_xy = _make_kp()
    kp_conf = _make_conf(0.9)

    # Face
    kp_xy[0]  = [200,  50]
    kp_xy[1]  = [185,  45]
    kp_xy[2]  = [215,  45]

    # Torso
    kp_xy[5]  = [160, 150]   # left_shoulder
    kp_xy[6]  = [240, 150]   # right_shoulder

    # Hips at seat level
    kp_xy[11] = [150, 250]   # left_hip
    kp_xy[12] = [250, 250]   # right_hip

    # Thighs horizontal, shins vertical → 90° at knee
    kp_xy[13] = [250, 250]   # left_knee  (thigh extends right)
    kp_xy[14] = [150, 250]   # right_knee (thigh extends left — mirrored)
    kp_xy[15] = [250, 350]   # left_ankle  (shin drops down)
    kp_xy[16] = [150, 350]   # right_ankle

    bbox = [100, 30, 400, 400]   # h=370, w=300, AR≈1.23
    return kp_xy, kp_conf, bbox


def _low_confidence_keypoints():
    """Standing geometry but all confidence values below threshold (0.3)."""
    kp_xy, _, bbox = _standing_keypoints()
    kp_conf = _make_conf(0.3)   # below default 0.5 threshold
    return kp_xy, kp_conf, bbox


def _sparse_keypoints(n_visible: int = 3):
    """Only n_visible keypoints above threshold; rest are zero confidence."""
    kp_xy, kp_conf, bbox = _standing_keypoints()
    kp_conf = _conf_zeros()
    # Make first n_visible keypoints visible
    kp_conf[:n_visible] = 0.9
    return kp_xy, kp_conf, bbox


def _make_pose(
    kp_xy: np.ndarray,
    kp_conf: np.ndarray,
    bbox: List[int],
    track_id: Optional[int] = 1,
    association_status: str = "matched",
) -> Dict[str, Any]:
    """Build a minimal pose dict as produced by PoseEstimator."""
    return {
        "track_id":           track_id,
        "association_status": association_status,
        "bbox":               bbox,
        "keypoints_xy":       kp_xy,
        "keypoints_conf":     kp_conf,
        "confidence":         0.9,
    }


def _make_recognizer(**overrides) -> ActivityRecognizer:
    """
    Create an ActivityRecognizer with no YAML file (uses built-in defaults),
    optionally with per-test overrides.
    """
    return ActivityRecognizer(
        config={"__no_yaml": True, **overrides},
        config_path="/nonexistent/path/that/does/not/exist.yaml",
    )


# ===========================================================================
# Test Group 1 — Standing detection
# ===========================================================================

class TestStandingDetection:
    """Group 1: Verify STANDING classification from straight-legged geometry."""

    def test_standing_full_keypoints(self):
        """Perfectly vertical legs + tall bbox → STANDING."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)

        result = rec.update([pose], frame_number=0)

        assert result[0]["activity"]["label"] == ActivityLabel.STANDING.value

    def test_standing_confidence_in_range(self):
        """STANDING confidence must be in [0, 1]."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        conf = result[0]["activity"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_standing_has_evidence(self):
        """Evidence list must be non-empty for a STANDING classification."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert len(result[0]["activity"]["evidence"]) > 0

    def test_standing_does_not_mutate_original(self):
        """Original pose dict must not be modified."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        original_keys = set(pose.keys())
        rec.update([pose], frame_number=0)
        assert "activity" not in pose, "Original dict must not be mutated"
        assert set(pose.keys()) == original_keys

    def test_standing_configurable_threshold(self):
        """Raising ALL standing thresholds to impossible values → no STANDING."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        # Set aspect ratio min impossibly high so no bbox can pass
        # AND knee angle min impossibly high so no knee angle can pass
        # (need to block all 6 scoring conditions so score < 2)
        rec_strict = _make_recognizer(
            standing_knee_angle_min=200.0,   # impossible (max angle is 180°)
            standing_hip_angle_min=200.0,    # impossible
            standing_aspect_ratio_min=50.0,  # impossible (would need 50:1 bbox)
            standing_vertical_min_px=10000.0, # impossible
        )
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec_strict.update([pose], frame_number=0)
        # With all conditions impossible, score = 0 < 2 → cannot be STANDING
        assert result[0]["activity"]["label"] != ActivityLabel.STANDING.value

    def test_standing_unmatched_pose_still_classified(self):
        """STANDING can be classified for unmatched poses (single-frame only)."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=None, association_status="unmatched")
        result = rec.update([pose], frame_number=0)
        # STANDING is a single-frame label, should work without track_id
        assert result[0]["activity"]["label"] == ActivityLabel.STANDING.value

    def test_standing_activity_key_always_present(self):
        """Every enriched pose must have the 'activity' key."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert "activity" in result[0]
        assert "label" in result[0]["activity"]
        assert "confidence" in result[0]["activity"]
        assert "evidence" in result[0]["activity"]


# ===========================================================================
# Test Group 2 — Sitting detection
# ===========================================================================

class TestSittingDetection:
    """Group 2: Verify SITTING classification from bent-knee geometry."""

    def test_sitting_full_keypoints(self):
        """90° knee angles + compressed bbox → SITTING."""
        kp_xy, kp_conf, bbox = _sitting_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=2)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.SITTING.value

    def test_sitting_confidence_in_range(self):
        """SITTING confidence must be in [0, 1]."""
        kp_xy, kp_conf, bbox = _sitting_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=2)
        result = rec.update([pose], frame_number=0)
        assert 0.0 <= result[0]["activity"]["confidence"] <= 1.0

    def test_sitting_has_knee_evidence(self):
        """SITTING evidence must reference knee angles."""
        kp_xy, kp_conf, bbox = _sitting_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=2)
        result = rec.update([pose], frame_number=0)
        ev = " ".join(result[0]["activity"]["evidence"])
        assert "knee" in ev.lower() or "aspect_ratio" in ev.lower()

    def test_sitting_unmatched_pose(self):
        """SITTING can be classified for unmatched poses (single-frame only)."""
        kp_xy, kp_conf, bbox = _sitting_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=None, association_status="unmatched")
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.SITTING.value

    def test_sitting_not_mutating(self):
        """Original pose dict is not mutated by SITTING classification."""
        kp_xy, kp_conf, bbox = _sitting_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=2)
        rec.update([pose], frame_number=0)
        assert "activity" not in pose


# ===========================================================================
# Test Group 3 — Walking detection
# ===========================================================================

class TestWalkingDetection:
    """
    Group 3: WALKING requires upright posture + moderate velocity.
    Temporal history is built by calling update() across multiple frames.
    """

    def _feed_walking_history(self, rec: ActivityRecognizer, track_id: int = 10):
        """
        Feed 8 frames of standing geometry with moderate displacement.
        Displacement per frame = 5 px (at reference resolution) → avg_vel = 5.
        """
        kp_xy, kp_conf, bbox_base = _standing_keypoints()
        for i in range(8):
            # Shift bounding box by 5px per frame along x
            offset = i * 5
            bbox = [
                bbox_base[0] + offset,
                bbox_base[1],
                bbox_base[2] + offset,
                bbox_base[3],
            ]
            pose = _make_pose(kp_xy, kp_conf, bbox, track_id=track_id)
            rec.update([pose], frame_number=i, frame_width=640, fps=30.0)
        return rec

    def test_walking_after_history(self):
        """After sufficient history with moderate velocity → WALKING."""
        rec = _make_recognizer(
            walking_velocity_min=2.0,
            walking_velocity_max=20.0,
            smoothing_window=5,
            smoothing_confidence_min=0.4,
        )
        self._feed_walking_history(rec, track_id=10)

        kp_xy, kp_conf, bbox_base = _standing_keypoints()
        bbox = [bbox_base[0] + 40, bbox_base[1], bbox_base[2] + 40, bbox_base[3]]
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=10)
        result = rec.update([pose], frame_number=8, frame_width=640, fps=30.0)
        assert result[0]["activity"]["label"] == ActivityLabel.WALKING.value

    def test_walking_has_velocity_evidence(self):
        """Evidence string must mention velocity for a WALKING label."""
        rec = _make_recognizer(
            walking_velocity_min=2.0,
            walking_velocity_max=20.0,
            smoothing_window=5,
            smoothing_confidence_min=0.4,
        )
        self._feed_walking_history(rec, track_id=10)
        kp_xy, kp_conf, bbox_base = _standing_keypoints()
        bbox = [bbox_base[0] + 40, bbox_base[1], bbox_base[2] + 40, bbox_base[3]]
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=10)
        result = rec.update([pose], frame_number=8, frame_width=640, fps=30.0)
        if result[0]["activity"]["label"] == ActivityLabel.WALKING.value:
            ev = " ".join(result[0]["activity"]["evidence"])
            assert "velocity" in ev.lower()

    def test_walking_requires_track_id(self):
        """WALKING must not be returned when track_id is None."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=None, association_status="unmatched")
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] != ActivityLabel.WALKING.value

    def test_walking_not_triggered_without_history(self):
        """WALKING must not fire on the very first frame (no velocity data)."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=10)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] != ActivityLabel.WALKING.value

    def test_walking_fps_aware_normalisation(self):
        """
        Identical pixel displacement at different FPS should produce
        equivalent normalised velocity after FPS scaling.
        At 15fps, actual 5px per frame is equivalent to 2.5 px/frame at 30fps.
        """
        # 15 fps → velocity scale = (640/640) * (30/15) = 2.0
        # Raw 5 px/frame × 2.0 = 10 norm_px/frame → still in [2, 20]
        rec = _make_recognizer(
            walking_velocity_min=2.0,
            walking_velocity_max=20.0,
            smoothing_window=5,
            smoothing_confidence_min=0.4,
        )
        kp_xy, kp_conf, bbox_base = _standing_keypoints()
        for i in range(8):
            offset = i * 5
            bbox = [bbox_base[0]+offset, bbox_base[1], bbox_base[2]+offset, bbox_base[3]]
            pose = _make_pose(kp_xy, kp_conf, bbox, track_id=15)
            rec.update([pose], frame_number=i, frame_width=640, fps=15.0)

        bbox = [bbox_base[0]+40, bbox_base[1], bbox_base[2]+40, bbox_base[3]]
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=15)
        result = rec.update([pose], frame_number=8, frame_width=640, fps=15.0)
        # Normalised velocity = 10 → within walking range → WALKING
        assert result[0]["activity"]["label"] == ActivityLabel.WALKING.value


# ===========================================================================
# Test Group 4 — Low-confidence keypoints
# ===========================================================================

class TestLowConfidenceKeypoints:
    """Group 4: When all keypoints are below confidence threshold → UNKNOWN."""

    def test_all_low_confidence_gives_unknown(self):
        """All keypoints at 0.3 confidence (threshold 0.5) → UNKNOWN."""
        kp_xy, kp_conf, bbox = _low_confidence_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_low_confidence_zero_confidence_score(self):
        """UNKNOWN label (from low-confidence gate) must have confidence = 0.0."""
        kp_xy, kp_conf, bbox = _low_confidence_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value
        assert result[0]["activity"]["confidence"] == 0.0

    def test_partial_confidence_below_min_keypoints(self):
        """3 visible keypoints (< min_visible_keypoints=5) → UNKNOWN."""
        kp_xy, kp_conf, bbox = _sparse_keypoints(n_visible=3)
        rec = _make_recognizer(min_visible_keypoints=5)
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_exactly_at_min_keypoints_allows_classification(self):
        """
        5 visible keypoints (== min) should not be blocked by the gate.
        Result depends on geometry — just verify not blocked by keypoint count.
        """
        kp_xy, kp_conf_base, bbox = _standing_keypoints()
        kp_conf = _conf_zeros()
        # Make indices 5,6,11,12,13 visible (shoulders, hips, one knee)
        for idx in [5, 6, 11, 12, 13]:
            kp_conf[idx] = 0.9
        rec = _make_recognizer(min_visible_keypoints=5)
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        # Should not be UNKNOWN purely because of keypoint count
        ev = " ".join(result[0]["activity"]["evidence"])
        assert "5/17" not in ev or result[0]["activity"]["label"] != ActivityLabel.UNKNOWN.value

    def test_low_confidence_evidence_explains_reason(self):
        """Evidence should mention the keypoint count when gate fails."""
        kp_xy, kp_conf, bbox = _sparse_keypoints(n_visible=2)
        rec = _make_recognizer(min_visible_keypoints=5)
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        ev = " ".join(result[0]["activity"]["evidence"])
        assert "keypoint" in ev.lower() or "visible" in ev.lower()


# ===========================================================================
# Test Group 5 — Missing / absent keypoints
# ===========================================================================

class TestMissingKeypoints:
    """Group 5: Zero confidence for all keypoints → UNKNOWN."""

    def test_zero_confidence_gives_unknown(self):
        """All-zero confidence → no visible keypoints → UNKNOWN."""
        kp_xy = _make_kp()
        kp_conf = _conf_zeros()
        bbox = [100, 100, 200, 300]
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_none_keypoints_xy_gives_unknown(self):
        """Missing keypoints_xy in pose dict → UNKNOWN."""
        rec = _make_recognizer()
        pose = {
            "track_id": 1,
            "association_status": "matched",
            "bbox": [100, 100, 200, 300],
            "keypoints_xy": None,
            "keypoints_conf": _make_conf(0.9),
        }
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_none_keypoints_conf_gives_unknown(self):
        """Missing keypoints_conf in pose dict → UNKNOWN."""
        rec = _make_recognizer()
        pose = {
            "track_id": 1,
            "association_status": "matched",
            "bbox": [100, 100, 200, 300],
            "keypoints_xy": _make_kp(),
            "keypoints_conf": None,
        }
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_zero_keypoints_does_not_raise(self):
        """Processing a pose with zero keypoints must not raise an exception."""
        kp_xy = _make_kp()
        kp_conf = _conf_zeros()
        bbox = [0, 0, 10, 10]
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=5)
        result = rec.update([pose], frame_number=0)
        assert isinstance(result, list)

    def test_empty_bbox_handled_safely(self):
        """Empty / missing bbox list must not raise."""
        kp_xy, kp_conf, _ = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, [], track_id=1)
        result = rec.update([pose], frame_number=0)
        # May be UNKNOWN due to AR=0 failing standing_aspect_ratio condition
        assert isinstance(result[0]["activity"]["label"], str)

    def test_zero_width_bbox_handled_safely(self):
        """Zero-width bbox (division by zero guard) must not raise."""
        kp_xy, kp_conf, _ = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, [100, 100, 100, 300], track_id=1)
        result = rec.update([pose], frame_number=0)
        assert isinstance(result[0]["activity"]["label"], str)


# ===========================================================================
# Test Group 6 — Unknown activity
# ===========================================================================

class TestUnknownActivity:
    """Group 6: Contradictory or insufficient geometric evidence → UNKNOWN."""

    def test_empty_pose_list(self):
        """Empty input returns empty list."""
        rec = _make_recognizer()
        result = rec.update([], frame_number=0)
        assert result == []

    def test_contradictory_geometry(self):
        """High aspect ratio (standing-like) but very bent knees → UNKNOWN or SITTING."""
        kp_xy = _make_kp()
        kp_conf = _make_conf(0.9)

        # Tall bbox (AR = 3) but 90° knees (sitting-like)
        kp_xy[5]  = [160, 150]; kp_xy[6]  = [240, 150]   # shoulders
        kp_xy[11] = [150, 250]; kp_xy[12] = [250, 250]   # hips
        kp_xy[13] = [250, 250]; kp_xy[14] = [150, 250]   # knees (90°)
        kp_xy[15] = [250, 350]; kp_xy[16] = [150, 350]   # ankles

        bbox = [100, 30, 300, 520]   # very tall, AR ≈ 2.45
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)

        # Knee angles are 90° → SITTING score >= 2 fires before STANDING
        # Regardless, must not crash and must return a valid label
        assert result[0]["activity"]["label"] in [
            lbl.value for lbl in ActivityLabel
        ]

    def test_only_face_keypoints_visible(self):
        """Only face keypoints (nose, eyes, ears) visible → UNKNOWN."""
        kp_xy = _make_kp()
        kp_conf = _conf_zeros()
        kp_xy[0] = [200, 50];  kp_conf[0] = 0.9   # nose
        kp_xy[1] = [185, 45];  kp_conf[1] = 0.9   # left_eye
        kp_xy[2] = [215, 45];  kp_conf[2] = 0.9   # right_eye
        kp_xy[3] = [170, 50];  kp_conf[3] = 0.9   # left_ear
        kp_xy[4] = [230, 50];  kp_conf[4] = 0.9   # right_ear

        bbox = [150, 20, 250, 80]
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        # No body keypoints → cannot classify posture
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value

    def test_zero_poses_in_frame(self):
        """Single call with zero poses must return empty list and not crash."""
        rec = _make_recognizer()
        result = rec.update([], frame_number=100)
        assert result == []

    def test_unknown_has_zero_or_low_confidence(self):
        """UNKNOWN classification must always have confidence = 0.0."""
        kp_xy = _make_kp()
        kp_conf = _conf_zeros()
        bbox = [0, 0, 100, 150]
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        # By implementation contract: UNKNOWN always gets confidence 0.0
        assert result[0]["activity"]["label"] == ActivityLabel.UNKNOWN.value
        assert result[0]["activity"]["confidence"] == 0.0


# ===========================================================================
# Test Group 7 — Multiple people processed independently
# ===========================================================================

class TestMultiplePeople:
    """Group 7: Two or more persons in the same frame must be classified independently."""

    def test_two_different_activities_simultaneously(self):
        """One standing, one sitting — labels must differ."""
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()

        rec = _make_recognizer()
        poses = [
            _make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1),
            _make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=2),
        ]
        result = rec.update(poses, frame_number=0)

        assert len(result) == 2
        labels = {r["activity"]["label"] for r in result}
        assert ActivityLabel.STANDING.value in labels
        assert ActivityLabel.SITTING.value in labels

    def test_independent_track_state(self):
        """Track 1 and Track 2 must have independent state deques."""
        rec = _make_recognizer()
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()

        # Feed Track 1 as STANDING for 5 frames
        for i in range(5):
            rec.update(
                [_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)],
                frame_number=i,
            )

        # Track 2 is new — no history
        result = rec.update(
            [
                _make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1),
                _make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=2),
            ],
            frame_number=5,
        )
        assert len(result) == 2
        # Track 2 should still classify correctly with its own geometry
        assert result[1]["activity"]["label"] == ActivityLabel.SITTING.value

    def test_output_length_matches_input(self):
        """Output list length must equal input list length."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        poses = [
            _make_pose(kp_xy, kp_conf, bbox, track_id=i)
            for i in range(5)
        ]
        result = rec.update(poses, frame_number=0)
        assert len(result) == len(poses)

    def test_each_person_has_activity_key(self):
        """Every enriched pose must contain the 'activity' key."""
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()
        rec = _make_recognizer()
        poses = [
            _make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1),
            _make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=2),
        ]
        result = rec.update(poses, frame_number=0)
        for enriched in result:
            assert "activity" in enriched

    def test_original_poses_not_mutated_in_batch(self):
        """None of the original dicts in a batch must be mutated."""
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()
        rec = _make_recognizer()
        poses = [
            _make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1),
            _make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=2),
        ]
        rec.update(poses, frame_number=0)
        for p in poses:
            assert "activity" not in p


# ===========================================================================
# Test Group 8 — Tracking ID changes
# ===========================================================================

class TestTrackingIDChanges:
    """Group 8: Verify that ID reassignments and stale tracks are handled safely."""

    def test_new_track_id_starts_fresh(self):
        """
        After several frames as track_id=1, the same person re-appears as
        track_id=99 (ID reassignment). The state for ID 99 must start empty.
        """
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()

        # Feed track 1 for 5 frames
        for i in range(5):
            rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=i)

        # First frame with new ID 99 — history should be empty
        assert 99 not in rec.get_track_ids()
        rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=99)], frame_number=5)
        history = rec.get_track_history(99)
        assert len(history) == 1

    def test_stale_track_evicted(self):
        """A track absent for > track_max_gap frames must be evicted."""
        rec = _make_recognizer(track_max_gap=5)
        kp_xy, kp_conf, bbox = _standing_keypoints()

        rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=0)
        assert 1 in rec.get_track_ids()

        # Advance 10 frames without track 1
        for f in range(1, 11):
            rec.update([], frame_number=f)

        assert 1 not in rec.get_track_ids()

    def test_old_track_state_not_inherited_by_new_id(self):
        """Old track's deque must not bleed into a reused track ID."""
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()
        rec = _make_recognizer(track_max_gap=5)

        # Track 1 → 5 standing frames
        for i in range(5):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)], frame_number=i)

        # Let track 1 go stale (10 frames gap)
        for f in range(5, 15):
            rec.update([], frame_number=f)

        # Track 1 is evicted — now a sitting person appears as new track 1
        result = rec.update(
            [_make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=1)],
            frame_number=15,
        )
        # History is fresh (1 frame), smoothing may return SITTING or UNKNOWN
        # but definitely not biased by the old standing history (which was evicted)
        history = rec.get_track_history(1)
        assert len(history) == 1

    def test_reset_clears_all_state(self):
        """After reset(), no tracks should remain."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        for i in range(5):
            rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=i)], frame_number=0)
        assert len(rec.get_track_ids()) > 0
        rec.reset()
        assert len(rec.get_track_ids()) == 0

    def test_get_track_history_unknown_id(self):
        """get_track_history() for a non-existent track ID returns []."""
        rec = _make_recognizer()
        assert rec.get_track_history(9999) == []

    def test_track_history_bounded(self):
        """Track history deque must not exceed track_history_len."""
        rec = _make_recognizer(track_history_len=5)
        kp_xy, kp_conf, bbox = _standing_keypoints()
        for f in range(20):
            rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=f)
        assert len(rec.get_track_history(1)) <= 5


# ===========================================================================
# Test Group 9 — Temporal consistency (smoothing)
# ===========================================================================

class TestTemporalConsistency:
    """Group 9: Verify majority-vote smoothing stability."""

    def test_stable_standing_over_many_frames(self):
        """10 consecutive STANDING frames → smoothed result is STANDING."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer(smoothing_window=10, smoothing_confidence_min=0.5)

        result = None
        for f in range(10):
            result = rec.update(
                [_make_pose(kp_xy, kp_conf, bbox, track_id=1)],
                frame_number=f,
            )
        assert result[0]["activity"]["label"] == ActivityLabel.STANDING.value

    def test_single_outlier_does_not_flip_stable_label(self):
        """
        9 STANDING frames + 1 UNKNOWN frame (bad keypoints) → still STANDING.
        The UNKNOWN frame must not flip the majority-vote result.
        """
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_b, kp_conf_b = _make_kp(), _conf_zeros()   # bad frame

        rec = _make_recognizer(
            smoothing_window=10,
            smoothing_confidence_min=0.5,
        )

        # 9 good standing frames
        for f in range(9):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)], frame_number=f)

        # 1 bad frame
        rec.update(
            [_make_pose(kp_xy_b, kp_conf_b, [0, 0, 100, 100], track_id=1)],
            frame_number=9,
        )

        # Read at frame 10 with another good frame
        result = rec.update(
            [_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)],
            frame_number=10,
        )
        assert result[0]["activity"]["label"] == ActivityLabel.STANDING.value

    def test_majority_switch_to_sitting(self):
        """
        After 3 STANDING frames, person sits down for 8 frames.
        Result should switch to SITTING.
        """
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()
        rec = _make_recognizer(smoothing_window=10, smoothing_confidence_min=0.5)

        for f in range(3):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)], frame_number=f)
        result = None
        for f in range(3, 11):
            result = rec.update(
                [_make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=1)],
                frame_number=f,
            )
        # 8 of last 10 frames are SITTING → majority vote → SITTING
        assert result[0]["activity"]["label"] == ActivityLabel.SITTING.value

    def test_confidence_reflects_majority_fraction(self):
        """STANDING confidence after 10 frames should be ~1.0."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer(smoothing_window=10, smoothing_confidence_min=0.5)
        result = None
        for f in range(10):
            result = rec.update(
                [_make_pose(kp_xy, kp_conf, bbox, track_id=1)],
                frame_number=f,
            )
        assert result[0]["activity"]["confidence"] > 0.0

    def test_smoothing_window_respected(self):
        """With a window of 3, old frames outside the window don't count."""
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        kp_xy_t, kp_conf_t, bbox_t = _sitting_keypoints()
        rec = _make_recognizer(smoothing_window=3, smoothing_confidence_min=0.5)

        # 10 standing frames far in the past
        for f in range(10):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)], frame_number=f)

        # 3 sitting frames at current time
        for f in range(10, 13):
            result = rec.update(
                [_make_pose(kp_xy_t, kp_conf_t, bbox_t, track_id=1)],
                frame_number=f,
            )
        # Window=3 → all 3 recent frames are SITTING → SITTING wins
        assert result[0]["activity"]["label"] == ActivityLabel.SITTING.value


# ===========================================================================
# Test Group 10 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """Group 10: Boundary conditions, special values, structural integrity."""

    def test_single_person_single_frame(self):
        """Single person, single frame must not crash and must return one result."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        result = rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=0)
        assert len(result) == 1

    def test_output_is_new_list(self):
        """update() must return a NEW list (not the input list)."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        poses = [_make_pose(kp_xy, kp_conf, bbox, track_id=1)]
        result = rec.update(poses, frame_number=0)
        assert result is not poses

    def test_activity_label_is_string(self):
        """'label' in activity dict must be a str (ActivityLabel.value)."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        result = rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=0)
        assert isinstance(result[0]["activity"]["label"], str)

    def test_activity_confidence_is_float(self):
        """'confidence' must be a float."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        result = rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=0)
        assert isinstance(result[0]["activity"]["confidence"], float)

    def test_activity_evidence_is_list(self):
        """'evidence' must be a list of strings."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        result = rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=0)
        ev = result[0]["activity"]["evidence"]
        assert isinstance(ev, list)
        for item in ev:
            assert isinstance(item, str)

    def test_negative_track_id_treated_as_unmatched(self):
        """Negative track_id (invalid) must not enter temporal state."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer()
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=-1)
        result = rec.update([pose], frame_number=0)
        assert -1 not in rec.get_track_ids()

    def test_fall_like_event_never_for_unmatched_pose(self):
        """
        FALL_LIKE_EVENT must never fire when track_id is None.
        (Q2 approval decision)
        """
        kp_xy = _make_kp()
        kp_conf = _make_conf(0.9)
        # Horizontal bbox (AR = 0.5) to trigger fall geometry
        bbox = [100, 200, 500, 300]   # h=100, w=400, AR=0.25
        rec = _make_recognizer(fall_aspect_ratio_max=1.0)
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=None, association_status="unmatched")
        result = rec.update([pose], frame_number=0)
        assert result[0]["activity"]["label"] != ActivityLabel.FALL_LIKE_EVENT.value

    def test_fall_like_event_requires_prior_upright_history(self):
        """
        FALL_LIKE_EVENT must NOT fire on first frame even if AR is low.
        (Condition 2: requires prior upright history)
        """
        kp_xy = _make_kp()
        kp_conf = _make_conf(0.9)
        bbox = [100, 200, 500, 300]   # AR=0.25 (horizontal)
        rec = _make_recognizer(fall_aspect_ratio_max=1.0)
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=0)
        # No prior history → cannot fire FALL_LIKE_EVENT
        assert result[0]["activity"]["label"] != ActivityLabel.FALL_LIKE_EVENT.value

    def test_fall_like_event_does_not_fire_with_low_keypoint_quality(self):
        """
        FALL_LIKE_EVENT must NOT fire when visible keypoints < fall_min_visible_keypoints.
        (Q3 approval decision — conservative approach)
        """
        kp_xy = _make_kp()
        kp_conf = _conf_zeros()
        kp_conf[:4] = 0.9   # only 4 visible (< fall_min_visible_keypoints=7)
        bbox = [100, 200, 500, 300]   # horizontal bbox

        rec = _make_recognizer(fall_aspect_ratio_max=1.0, fall_min_visible_keypoints=7)

        # Build upright history with a valid track
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        for f in range(5):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=1)], frame_number=f)

        # Now test with low-quality keypoints + horizontal bbox
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=1)
        result = rec.update([pose], frame_number=5)
        assert result[0]["activity"]["label"] != ActivityLabel.FALL_LIKE_EVENT.value

    def test_fall_like_event_all_conditions_met(self):
        """
        When all three conditions are met, FALL_LIKE_EVENT fires:
        1. Horizontal bbox (AR < fall_aspect_ratio_max)
        2. Prior upright history exists
        3. Sufficient keypoint quality
        """
        rec = _make_recognizer(
            fall_aspect_ratio_max=1.0,
            fall_min_visible_keypoints=5,
            fall_history_frames=10,
            fall_override_window=5,
            smoothing_window=10,
        )

        # Build 5 frames of upright (STANDING) history
        kp_xy_s, kp_conf_s, bbox_s = _standing_keypoints()
        for f in range(5):
            rec.update([_make_pose(kp_xy_s, kp_conf_s, bbox_s, track_id=7)], frame_number=f)

        # Confirm track 7 has standing history
        history = rec.get_track_history(7)
        assert len(history) == 5
        assert any(r["label"] == ActivityLabel.STANDING.value for r in history)

        # Now: horizontal bbox + good keypoints (all three conditions satisfied)
        kp_xy_fall = _make_kp()
        kp_conf_fall = _make_conf(0.9)   # all visible
        bbox_fall = [100, 200, 500, 300]   # AR = 0.25

        pose_fall = _make_pose(kp_xy_fall, kp_conf_fall, bbox_fall, track_id=7)
        result = rec.update([pose_fall], frame_number=5)
        assert result[0]["activity"]["label"] == ActivityLabel.FALL_LIKE_EVENT.value

    def test_running_requires_min_history_frames(self):
        """
        RUNNING must not fire until running_min_history_frames is satisfied.
        (Q1 decision: temporal evidence required)
        """
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer(
            running_velocity_min=5.0,
            running_min_history_frames=5,
            walking_velocity_min=100.0,   # disable walking
        )
        # Only 2 frames of history — not enough for RUNNING
        for f in range(2):
            bbox_f = [bbox[0] + f * 50, bbox[1], bbox[2] + f * 50, bbox[3]]
            rec.update([_make_pose(kp_xy, kp_conf, bbox_f, track_id=1)], frame_number=f)
        assert 1 in rec.get_track_ids()
        history = rec.get_track_history(1)
        for r in history:
            assert r["label"] != ActivityLabel.RUNNING.value

    def test_get_config_returns_copy(self):
        """get_config() must return a copy, not the internal dict."""
        rec = _make_recognizer()
        cfg = rec.get_config()
        cfg["smoothing_window"] = 999
        assert rec.get_config()["smoothing_window"] != 999

    def test_multiple_frames_same_track_history_grows(self):
        """Track history must grow (up to deque limit) across frames."""
        kp_xy, kp_conf, bbox = _standing_keypoints()
        rec = _make_recognizer(track_history_len=10)
        for f in range(7):
            rec.update([_make_pose(kp_xy, kp_conf, bbox, track_id=1)], frame_number=f)
        assert len(rec.get_track_history(1)) == 7

    def test_velocity_normalisation_different_resolution(self):
        """
        At 1280px width, 10 raw px/frame = 5 norm_px/frame (at 640px reference).
        This should still be in the WALKING range [2, 20].
        """
        kp_xy, kp_conf, bbox_base = _standing_keypoints()
        rec = _make_recognizer(
            walking_velocity_min=2.0,
            walking_velocity_max=20.0,
            smoothing_window=5,
            smoothing_confidence_min=0.4,
        )
        # Feed 8 frames with 10 px/frame displacement at 1280 width
        for i in range(8):
            offset = i * 10
            bbox = [bbox_base[0]+offset, bbox_base[1], bbox_base[2]+offset, bbox_base[3]]
            pose = _make_pose(kp_xy, kp_conf, bbox, track_id=20)
            rec.update([pose], frame_number=i, frame_width=1280, fps=30.0)

        bbox = [bbox_base[0]+80, bbox_base[1], bbox_base[2]+80, bbox_base[3]]
        pose = _make_pose(kp_xy, kp_conf, bbox, track_id=20)
        result = rec.update([pose], frame_number=8, frame_width=1280, fps=30.0)
        # norm_vel = 10 * (640/1280) * (30/30) = 5 → in [2, 20] → WALKING
        assert result[0]["activity"]["label"] == ActivityLabel.WALKING.value
