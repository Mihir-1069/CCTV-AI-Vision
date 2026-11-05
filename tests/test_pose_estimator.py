"""
Unit tests for Pose Estimation module.

Tests are split into:
- Mocked unit tests (no model download required, deterministic)
- Model smoke tests (actual YOLO11n-pose loading and CPU inference)

Mocked tests use a MockPoseResult helper that mimics the Ultralytics
result structure without requiring model weights.
"""

import pytest
import numpy as np
import cv2
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.pose.pose_utils import PoseUtils
from ai_engine.pose.pose_analyzer import PoseAnalyzer
from ai_engine.pose.pose_estimator import PoseEstimator


# ---------------------------------------------------------------------------
# Mock helpers — simulate Ultralytics YOLO pose output
# ---------------------------------------------------------------------------

class MockTensor:
    """Simulates a torch-like tensor with .cpu().numpy() chain."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class MockBoxes:
    """Simulates ultralytics.engine.results.Boxes."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = MockTensor(xyxy)
        self.conf = MockTensor(conf)
        self.cls = MockTensor(cls)

    def __len__(self):
        return len(self.conf.numpy())


class MockKeypoints:
    """Simulates ultralytics.engine.results.Keypoints."""

    def __init__(self, xy, conf):
        self.xy = MockTensor(xy)
        self.conf = MockTensor(conf)


class MockResult:
    """Simulates ultralytics.engine.results.Results for pose task."""

    def __init__(self, boxes_xyxy, boxes_conf, boxes_cls, kpts_xy, kpts_conf):
        if len(boxes_xyxy) > 0:
            self.boxes = MockBoxes(boxes_xyxy, boxes_conf, boxes_cls)
        else:
            self.boxes = MockBoxes(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
        self.keypoints = MockKeypoints(kpts_xy, kpts_conf)
        self.speed = {"preprocess": 1.0, "inference": 100.0, "postprocess": 1.0}
        self.names = {0: "person"}


def make_single_person_result(
    bbox=(100, 50, 300, 450),
    confidence=0.92,
    keypoint_base_conf=0.8
):
    """
    Create a mock result for one person with valid keypoints.
    """
    xyxy = np.array([list(bbox)])
    conf = np.array([confidence])
    cls = np.array([0])

    # 17 keypoints — simple grid pattern
    kpts_xy = np.zeros((1, 17, 2))
    kpts_conf = np.zeros((1, 17))

    for i in range(17):
        kpts_xy[0, i] = [bbox[0] + 50 + i * 5, bbox[1] + 50 + i * 5]
        kpts_conf[0, i] = keypoint_base_conf + i * 0.01
        if kpts_conf[0, i] > 1.0:
            kpts_conf[0, i] = 1.0

    return MockResult(xyxy, conf, cls, kpts_xy, kpts_conf)


def make_two_person_result():
    """Create a mock result for two persons."""
    xyxy = np.array([
        [100, 50, 300, 450],
        [350, 50, 550, 450]
    ])
    conf = np.array([0.92, 0.85])
    cls = np.array([0, 0])

    kpts_xy = np.zeros((2, 17, 2))
    kpts_conf = np.zeros((2, 17))

    for p in range(2):
        for i in range(17):
            kpts_xy[p, i] = [xyxy[p, 0] + 50 + i * 5, xyxy[p, 1] + 50 + i * 5]
            kpts_conf[p, i] = 0.7 + i * 0.01
            if kpts_conf[p, i] > 1.0:
                kpts_conf[p, i] = 1.0

    return MockResult(xyxy, conf, cls, kpts_xy, kpts_conf)


def make_empty_result():
    """Create a mock result with no detections."""
    return MockResult(
        np.zeros((0, 4)),
        np.zeros((0,)),
        np.zeros((0,)),
        np.zeros((0, 17, 2)),
        np.zeros((0, 17))
    )


def make_partial_keypoints_result():
    """
    Create a mock result where some keypoints have low confidence.
    """
    xyxy = np.array([[100, 50, 300, 450]])
    conf = np.array([0.88])
    cls = np.array([0])

    kpts_xy = np.zeros((1, 17, 2))
    kpts_conf = np.zeros((1, 17))

    for i in range(17):
        kpts_xy[0, i] = [100 + i * 10, 50 + i * 10]
        # First 10 keypoints above threshold, rest below
        kpts_conf[0, i] = 0.8 if i < 10 else 0.2

    return MockResult(xyxy, conf, cls, kpts_xy, kpts_conf)


def patch_yolo_pose(mock_result):
    """
    Patch the YOLO class so that model(frame, ...) returns [mock_result].
    """
    mock_model = Mock()
    mock_model.return_value = [mock_result]

    patcher = patch('ai_engine.pose.pose_estimator.YOLO', return_value=mock_model)
    return patcher, mock_model


# ---------------------------------------------------------------------------
# Test: PoseUtils
# ---------------------------------------------------------------------------

class TestPoseUtils:
    """Test cases for PoseUtils class."""

    def test_coco_keypoint_names_count(self):
        """Verify 17 COCO keypoint names."""
        assert len(PoseUtils.COCO_KEYPOINT_NAMES) == 17

    def test_coco_keypoint_names_order(self):
        """Verify correct ordering of COCO keypoint names."""
        assert PoseUtils.COCO_KEYPOINT_NAMES[0] == "nose"
        assert PoseUtils.COCO_KEYPOINT_NAMES[5] == "left_shoulder"
        assert PoseUtils.COCO_KEYPOINT_NAMES[6] == "right_shoulder"
        assert PoseUtils.COCO_KEYPOINT_NAMES[11] == "left_hip"
        assert PoseUtils.COCO_KEYPOINT_NAMES[16] == "right_ankle"

    def test_skeleton_connections_count(self):
        """Verify skeleton connections exist."""
        assert len(PoseUtils.SKELETON_CONNECTIONS) > 0
        # Each connection is a pair of valid indices
        for a, b in PoseUtils.SKELETON_CONNECTIONS:
            assert 0 <= a < 17
            assert 0 <= b < 17
            assert a != b

    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        iou = PoseUtils.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert iou == 0.0

    def test_calculate_iou_full_overlap(self):
        """Test IoU with full overlap."""
        iou = PoseUtils.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert iou == 1.0

    def test_calculate_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        iou = PoseUtils.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.14 < iou < 0.15

    def test_keypoints_to_dict_all_visible(self):
        """Test converting keypoints with all above threshold."""
        kp_xy = np.array([[i * 10, i * 10] for i in range(17)], dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)

        result = PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5)

        assert len(result) == 17
        for name in PoseUtils.COCO_KEYPOINT_NAMES:
            assert result[name] is not None
            assert "x" in result[name]
            assert "y" in result[name]
            assert "confidence" in result[name]

    def test_keypoints_to_dict_some_below_threshold(self):
        """Test that low-confidence keypoints become None."""
        kp_xy = np.array([[i * 10, i * 10] for i in range(17)], dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)
        kp_conf[5] = 0.2  # left_shoulder below threshold

        result = PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5)

        assert result["left_shoulder"] is None
        assert result["nose"] is not None

    def test_keypoints_to_dict_all_below_threshold(self):
        """Test all keypoints below threshold."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.1, dtype=np.float32)

        result = PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5)

        for name in PoseUtils.COCO_KEYPOINT_NAMES:
            assert result[name] is None

    def test_filter_keypoints(self):
        """Test filtering keypoints by confidence."""
        kp_xy = np.array([[i * 10, i * 10] for i in range(17)], dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)
        kp_conf[3] = 0.1
        kp_conf[7] = 0.1

        filtered = PoseUtils.filter_keypoints(kp_xy, kp_conf, threshold=0.5)

        assert len(filtered) == 15
        names = [f["name"] for f in filtered]
        assert "left_ear" not in names
        assert "left_elbow" not in names
        assert "nose" in names

    def test_draw_keypoints(self):
        """Test drawing keypoints on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp_xy = np.array([[100, 100], [200, 200]] + [[0, 0]] * 15, dtype=np.float32)
        kp_conf = np.array([0.9, 0.8] + [0.1] * 15, dtype=np.float32)

        result = PoseUtils.draw_keypoints(frame, kp_xy, kp_conf, threshold=0.5)

        assert result.shape == frame.shape
        # Original should not be modified
        assert not np.array_equal(frame, result)

    def test_draw_skeleton(self):
        """Test drawing skeleton on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)
        for i in range(17):
            kp_xy[i] = [50 + i * 10, 50 + i * 10]

        result = PoseUtils.draw_skeleton(frame, kp_xy, kp_conf, threshold=0.5)

        assert result.shape == frame.shape
        assert not np.array_equal(frame, result)

    def test_draw_skeleton_skips_low_confidence(self):
        """Test skeleton drawing skips connections with low-confidence endpoints."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)
        kp_conf[0] = 0.1  # nose below threshold

        result = PoseUtils.draw_skeleton(frame, kp_xy, kp_conf, threshold=0.5)

        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Test: PoseEstimator (mocked)
# ---------------------------------------------------------------------------

class TestPoseEstimatorInit:
    """Test PoseEstimator initialization and configuration."""

    @pytest.fixture
    def estimator_config(self):
        return {
            "model_name": "yolo11n-pose.pt",
            "device": "cpu",
            "confidence_threshold": 0.5,
            "keypoint_threshold": 0.5,
            "image_size": 640,
            "iou_threshold": 0.45,
            "verbose": False,
        }

    def test_init_with_config(self, estimator_config):
        """Test initialization with explicit config."""
        estimator = PoseEstimator(config=estimator_config)

        assert estimator.model_name == "yolo11n-pose.pt"
        assert estimator.device == "cpu"
        assert estimator.confidence_threshold == 0.5
        assert estimator.keypoint_threshold == 0.5
        assert estimator.image_size == 640
        assert estimator.iou_threshold == 0.45
        assert estimator.verbose is False

    def test_init_with_config_file(self):
        """Test initialization from model.yaml."""
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
        estimator = PoseEstimator(config_path=str(config_path))

        assert estimator.model_name == "yolo11n-pose.pt"
        assert estimator.device == "cpu"
        assert estimator.image_size == 640
        assert estimator.association_iou_threshold == 0.5

    def test_init_defaults(self):
        """Test initialization with no config (defaults)."""
        estimator = PoseEstimator()

        assert estimator.model_name == "yolo11n-pose.pt"
        assert estimator.device == "cpu"

    def test_get_model_info(self, estimator_config):
        """Test getting model info."""
        estimator = PoseEstimator(config=estimator_config)
        info = estimator.get_model_info()

        assert info["model_name"] == "yolo11n-pose.pt"
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.5
        assert info["keypoint_threshold"] == 0.5
        assert info["image_size"] == 640
        assert info["is_loaded"] is False
        assert info["keypoint_count"] == 17
        assert len(info["keypoint_names"]) == 17

    def test_get_device(self, estimator_config):
        """Test getting device."""
        estimator = PoseEstimator(config=estimator_config)
        assert estimator.get_device() == "cpu"

    def test_repr(self, estimator_config):
        """Test string representation."""
        estimator = PoseEstimator(config=estimator_config)
        repr_str = repr(estimator)

        assert "PoseEstimator" in repr_str
        assert "yolo11n-pose.pt" in repr_str
        assert "not loaded" in repr_str


class TestPoseEstimatorPerformance:
    """Test performance tracking."""

    def test_performance_stats_empty(self):
        """Test performance stats with no inferences."""
        estimator = PoseEstimator(config={"verbose": False})
        stats = estimator.get_performance_stats()

        assert stats["inference_count"] == 0
        assert stats["avg_latency_ms"] is None

    def test_reset_performance_stats(self):
        """Test resetting performance stats."""
        estimator = PoseEstimator(config={"verbose": False})
        estimator._inference_times = [0.1, 0.2, 0.3]
        estimator.reset_performance_stats()

        assert estimator._inference_times == []


class TestPoseEstimatorEstimate:
    """Test pose estimation with mocked YOLO."""

    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_estimate_without_model(self, sample_frame):
        """Test estimate without loading model."""
        estimator = PoseEstimator(config={"verbose": False})
        result = estimator.estimate(sample_frame)

        assert result == []

    def test_estimate_none_frame(self):
        """Test estimate with None frame."""
        estimator = PoseEstimator(config={"verbose": False})
        estimator.model = Mock()
        result = estimator.estimate(None)

        assert result == []

    def test_estimate_non_array_frame(self):
        """Test estimate with non-array frame."""
        estimator = PoseEstimator(config={"verbose": False})
        estimator.model = Mock()
        result = estimator.estimate("not_an_array")

        assert result == []

    def test_estimate_invalid_shape(self):
        """Test estimate with invalid frame shape."""
        estimator = PoseEstimator(config={"verbose": False})
        estimator.model = Mock()
        result = estimator.estimate(np.array([1, 2, 3]))

        assert result == []

    def test_estimate_empty_detections(self, sample_frame):
        """Test estimate with no persons detected."""
        mock_result = make_empty_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            assert result == []
        finally:
            patcher.stop()

    def test_estimate_single_person(self, sample_frame):
        """Test estimate with one person."""
        mock_result = make_single_person_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            assert len(result) == 1
            pose = result[0]

            assert "bbox" in pose
            assert "confidence" in pose
            assert "keypoints_xy" in pose
            assert "keypoints_conf" in pose
            assert "keypoints" in pose

            assert len(pose["bbox"]) == 4
            assert isinstance(pose["confidence"], float)
            assert pose["keypoints_xy"].shape == (17, 2)
            assert pose["keypoints_conf"].shape == (17,)
        finally:
            patcher.stop()

    def test_estimate_two_persons(self, sample_frame):
        """Test estimate with two persons."""
        mock_result = make_two_person_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            assert len(result) == 2

            # Each pose should have full keypoint structure
            for pose in result:
                assert pose["keypoints_xy"].shape == (17, 2)
                assert pose["keypoints_conf"].shape == (17,)
        finally:
            patcher.stop()

    def test_estimate_bbox_clipped(self, sample_frame):
        """Test that bounding boxes are clipped to frame boundaries."""
        # Bbox extends beyond frame
        bbox = (100, 50, 1000, 1000)
        mock_result = make_single_person_result(bbox=bbox)
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            assert len(result) == 1
            bbox_result = result[0]["bbox"]
            # Frame is 640x480
            assert bbox_result[2] <= 640
            assert bbox_result[3] <= 480
        finally:
            patcher.stop()

    def test_estimate_keypoint_mapping(self, sample_frame):
        """Test that keypoints are correctly mapped to COCO names."""
        mock_result = make_single_person_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False, "keypoint_threshold": 0.0})
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            pose = result[0]
            kp_dict = pose["keypoints"]

            # All 17 names should be present
            for name in PoseUtils.COCO_KEYPOINT_NAMES:
                assert name in kp_dict
                assert kp_dict[name] is not None  # all above 0 threshold
                assert "x" in kp_dict[name]
                assert "y" in kp_dict[name]
                assert "confidence" in kp_dict[name]
        finally:
            patcher.stop()

    def test_estimate_low_confidence_keypoints_filtered(self, sample_frame):
        """Test that low-confidence keypoints are None in output."""
        mock_result = make_partial_keypoints_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={
                "verbose": False,
                "keypoint_threshold": 0.5
            })
            estimator.model = mock_model
            result = estimator.estimate(sample_frame)

            pose = result[0]
            kp_dict = pose["keypoints"]

            # First 10 keypoints above threshold (0.8), rest below (0.2)
            for i, name in enumerate(PoseUtils.COCO_KEYPOINT_NAMES):
                if i < 10:
                    assert kp_dict[name] is not None
                else:
                    assert kp_dict[name] is None
        finally:
            patcher.stop()

    def test_estimate_performance_tracked(self, sample_frame):
        """Test that inference time is tracked."""
        mock_result = make_single_person_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model

            for _ in range(5):
                estimator.estimate(sample_frame)

            stats = estimator.get_performance_stats()
            assert stats["inference_count"] == 5
            assert stats["avg_latency_ms"] > 0
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Test: Pose Estimator drawing
# ---------------------------------------------------------------------------

class TestPoseEstimatorDrawing:
    """Test drawing methods."""

    @pytest.fixture
    def sample_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_pose(self):
        return {
            "bbox": [100, 50, 300, 450],
            "confidence": 0.9,
            "keypoints_xy": np.zeros((17, 2), dtype=np.float32),
            "keypoints_conf": np.full(17, 0.9, dtype=np.float32),
            "keypoints": {},
            "track_id": 1,
        }

    def test_draw_poses(self, sample_frame, sample_pose):
        """Test drawing poses on frame."""
        estimator = PoseEstimator(config={"verbose": False})
        result = estimator.draw_poses(sample_frame, [sample_pose])

        assert result.shape == sample_frame.shape
        assert not np.array_equal(sample_frame, result)

    def test_draw_poses_empty(self, sample_frame):
        """Test drawing with no poses."""
        estimator = PoseEstimator(config={"verbose": False})
        result = estimator.draw_poses(sample_frame, [])

        assert np.array_equal(sample_frame, result)

    def test_draw_poses_no_track_id(self, sample_frame, sample_pose):
        """Test drawing pose without track_id."""
        estimator = PoseEstimator(config={"verbose": False})
        pose = dict(sample_pose)
        pose["track_id"] = None
        result = estimator.draw_poses(sample_frame, [pose])

        assert result.shape == sample_frame.shape


# ---------------------------------------------------------------------------
# Test: Tracking association
# ---------------------------------------------------------------------------

class TestTrackingAssociation:
    """Test pose-to-track IoU association."""

    def test_associate_perfect_match(self):
        """Test perfect IoU match."""
        poses = [{"bbox": [100, 100, 200, 200], "confidence": 0.9}]
        tracks = [{"track_id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9}]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.5)

        assert result[0]["track_id"] == 1
        assert result[0]["association_status"] == "matched"

    def test_associate_no_match(self):
        """Test when no track has sufficient IoU."""
        poses = [{"bbox": [0, 0, 100, 100], "confidence": 0.9}]
        tracks = [{"track_id": 1, "bbox": [500, 500, 600, 600], "confidence": 0.9}]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.5)

        assert result[0]["track_id"] is None
        assert result[0]["association_status"] == "unmatched"

    def test_associate_empty_poses(self):
        """Test with no poses."""
        result = PoseUtils.associate_poses_to_tracks([], [{"track_id": 1, "bbox": [0, 0, 10, 10]}])
        assert result == []

    def test_associate_empty_tracks(self):
        """Test with no tracks."""
        poses = [{"bbox": [0, 0, 100, 100]}]
        result = PoseUtils.associate_poses_to_tracks(poses, [], iou_threshold=0.5)

        assert result[0]["track_id"] is None
        assert result[0]["association_status"] == "unmatched"

    def test_one_to_one_assignment(self):
        """Test that one-to-one constraint is enforced."""
        # Two poses, one track — both poses overlap with the same track
        poses = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"bbox": [105, 105, 205, 205], "confidence": 0.85},
        ]
        tracks = [
            {"track_id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9},
        ]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.3)

        # Only one pose should be matched (the one with higher IoU)
        matched = [r for r in result if r["association_status"] == "matched"]
        assert len(matched) == 1
        assert matched[0]["track_id"] == 1

        # The other should be uncertain (its best candidate was taken)
        uncertain = [r for r in result if r["association_status"] == "uncertain"]
        assert len(uncertain) == 1
        assert uncertain[0]["track_id"] is None

    def test_track_id_not_duplicated(self):
        """Test that a track_id is never assigned to multiple poses."""
        poses = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"bbox": [110, 110, 210, 210], "confidence": 0.85},
            {"bbox": [120, 120, 220, 220], "confidence": 0.8},
        ]
        tracks = [
            {"track_id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9},
        ]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.2)

        matched = [r for r in result if r["track_id"] is not None]
        assert len(matched) == 1  # Only one pose gets track_id 1

    def test_multiple_poses_multiple_tracks(self):
        """Test matching multiple poses to multiple tracks."""
        poses = [
            {"bbox": [0, 0, 100, 100], "confidence": 0.9},
            {"bbox": [200, 200, 300, 300], "confidence": 0.9},
        ]
        tracks = [
            {"track_id": 1, "bbox": [0, 0, 100, 100], "confidence": 0.9},
            {"track_id": 2, "bbox": [200, 200, 300, 300], "confidence": 0.9},
        ]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.5)

        assert result[0]["track_id"] == 1
        assert result[0]["association_status"] == "matched"
        assert result[1]["track_id"] == 2
        assert result[1]["association_status"] == "matched"

    def test_greedy_optimal_assignment(self):
        """Test that greedy picks the best IoU first."""
        # Pose 0 overlaps more with track 0
        # Pose 1 overlaps slightly with track 0 but also with track 1
        poses = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"bbox": [180, 180, 280, 280], "confidence": 0.9},
        ]
        tracks = [
            {"track_id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"track_id": 2, "bbox": [190, 190, 290, 290], "confidence": 0.9},
        ]

        result = PoseUtils.associate_poses_to_tracks(poses, tracks, iou_threshold=0.1)

        # Pose 0 should match track 1 (perfect overlap)
        assert result[0]["track_id"] == 1
        # Pose 1 should match track 2 (partial overlap)
        assert result[1]["track_id"] == 2


# ---------------------------------------------------------------------------
# Test: JSON serialization
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    """Test that pose output can be serialized to JSON."""

    def test_pose_result_json_serializable(self):
        """Test full pose result JSON serialization."""
        kp_xy = np.array([[i * 10, i * 10 + 5] for i in range(17)], dtype=np.float32)
        kp_conf = np.full(17, 0.85, dtype=np.float32)

        keypoints_dict = PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5)

        pose_result = {
            "track_id": 1,
            "association_status": "matched",
            "pose_confidence": 0.92,
            "bbox": [100, 50, 300, 450],
            "keypoints": keypoints_dict,
        }

        # Should not raise
        json_str = json.dumps(pose_result, indent=2)
        parsed = json.loads(json_str)

        assert parsed["track_id"] == 1
        assert parsed["bbox"] == [100, 50, 300, 450]
        assert parsed["keypoints"]["nose"]["x"] == 0.0
        assert parsed["keypoints"]["nose"]["confidence"] == pytest.approx(0.85, abs=0.01)

    def test_pose_result_with_null_keypoints(self):
        """Test JSON serialization with null keypoints."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.2, dtype=np.float32)

        keypoints_dict = PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5)

        pose_result = {
            "track_id": None,
            "association_status": "unmatched",
            "pose_confidence": 0.6,
            "bbox": [100, 50, 300, 450],
            "keypoints": keypoints_dict,
        }

        json_str = json.dumps(pose_result)
        parsed = json.loads(json_str)

        assert parsed["track_id"] is None
        assert parsed["keypoints"]["nose"] is None

    def test_full_frame_output_json(self):
        """Test full frame output JSON structure."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.8, dtype=np.float32)
        kp_xy[0] = [200, 70]
        kp_xy[5] = [160, 150]

        pose = {
            "track_id": 1,
            "association_status": "matched",
            "pose_confidence": 0.91,
            "bbox": [100, 50, 300, 450],
            "keypoints": PoseUtils.keypoints_to_dict(kp_xy, kp_conf, threshold=0.5),
        }

        frame_output = {
            "camera_id": "cam_001",
            "frame_number": 100,
            "poses": [pose],
        }

        json_str = json.dumps(frame_output, indent=2)
        parsed = json.loads(json_str)

        assert parsed["camera_id"] == "cam_001"
        assert parsed["frame_number"] == 100
        assert len(parsed["poses"]) == 1
        assert parsed["poses"][0]["keypoints"]["nose"]["x"] == 200.0


# ---------------------------------------------------------------------------
# Test: PoseAnalyzer
# ---------------------------------------------------------------------------

class TestPoseAnalyzer:
    """Test cases for PoseAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return PoseAnalyzer(keypoint_threshold=0.5)

    @pytest.fixture
    def full_keypoints(self):
        """Keypoints for a person with all joints visible."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.9, dtype=np.float32)

        # Place keypoints in a realistic layout
        kp_xy[0] = [200, 80]    # nose
        kp_xy[5] = [160, 150]   # left_shoulder
        kp_xy[6] = [240, 150]   # right_shoulder
        kp_xy[7] = [140, 220]   # left_elbow
        kp_xy[8] = [260, 220]   # right_elbow
        kp_xy[9] = [130, 290]   # left_wrist
        kp_xy[10] = [270, 290]  # right_wrist
        kp_xy[11] = [170, 300]  # left_hip
        kp_xy[12] = [230, 300]  # right_hip
        kp_xy[13] = [170, 380]  # left_knee
        kp_xy[14] = [230, 380]  # right_knee
        kp_xy[15] = [170, 450]  # left_ankle
        kp_xy[16] = [230, 450]  # right_ankle

        return kp_xy, kp_conf

    def test_analyzer_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.keypoint_threshold == 0.5

    def test_analyze_full_keypoints(self, analyzer, full_keypoints):
        """Test analysis with all keypoints visible."""
        kp_xy, kp_conf = full_keypoints
        result = analyzer.analyze(kp_xy, kp_conf)

        assert "joint_angles" in result
        assert "body_orientation" in result
        assert "visibility_stats" in result

        # Check angles exist
        angles = result["joint_angles"]
        assert "left_elbow" in angles
        assert "right_elbow" in angles
        assert "left_knee" in angles
        assert "right_knee" in angles
        assert "left_hip" in angles
        assert "right_hip" in angles
        assert "left_shoulder" in angles
        assert "right_shoulder" in angles

        # Angles should be valid numbers
        for angle_name, angle_val in angles.items():
            assert angle_val is not None, f"{angle_name} should not be None with full keypoints"
            assert 0 <= angle_val <= 180

    def test_analyze_missing_keypoints(self, analyzer):
        """Test analysis with missing keypoints."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.zeros(17, dtype=np.float32)

        # Only shoulders visible
        kp_xy[5] = [160, 150]
        kp_conf[5] = 0.9
        kp_xy[6] = [240, 150]
        kp_conf[6] = 0.9

        result = analyzer.analyze(kp_xy, kp_conf)

        # Elbow angles should be None (no elbow/wrist keypoints)
        assert result["joint_angles"]["left_elbow"] is None
        assert result["joint_angles"]["right_elbow"] is None
        # Knee angles should be None
        assert result["joint_angles"]["left_knee"] is None
        assert result["joint_angles"]["right_knee"] is None

    def test_analyze_all_missing(self, analyzer):
        """Test analysis with all keypoints missing."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.zeros(17, dtype=np.float32)

        result = analyzer.analyze(kp_xy, kp_conf)

        for angle_val in result["joint_angles"].values():
            assert angle_val is None

    def test_visibility_stats(self, analyzer, full_keypoints):
        """Test visibility statistics."""
        kp_xy, kp_conf = full_keypoints
        result = analyzer.analyze(kp_xy, kp_conf)

        stats = result["visibility_stats"]
        assert stats["visible_keypoints"] == 17
        assert stats["total_keypoints"] == 17
        assert stats["visibility_ratio"] == 1.0
        assert stats["mean_confidence"] > 0

    def test_visibility_stats_partial(self, analyzer):
        """Test visibility stats with partial keypoints."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.full(17, 0.2, dtype=np.float32)
        kp_conf[:5] = 0.9  # Only 5 visible

        result = analyzer.analyze(kp_xy, kp_conf)
        stats = result["visibility_stats"]

        assert stats["visible_keypoints"] == 5
        assert stats["visibility_ratio"] == round(5 / 17, 4)

    def test_body_orientation_front(self, analyzer, full_keypoints):
        """Test body orientation detection (front facing)."""
        kp_xy, kp_conf = full_keypoints
        result = analyzer.analyze(kp_xy, kp_conf)

        # With nose and both eyes visible, should be "front"
        assert result["body_orientation"] in ["front", "unknown"]

    def test_body_orientation_unknown(self, analyzer):
        """Test body orientation with insufficient keypoints."""
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_conf = np.zeros(17, dtype=np.float32)

        result = analyzer.analyze(kp_xy, kp_conf)
        assert result["body_orientation"] == "unknown"


# ---------------------------------------------------------------------------
# Test: Estimate with tracking integration
# ---------------------------------------------------------------------------

class TestEstimateWithTracking:
    """Test estimate_with_tracking method."""

    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_with_matching_track(self, sample_frame):
        """Test pose estimation with matching tracking output."""
        mock_result = make_single_person_result(bbox=(100, 100, 200, 200))
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model

            tracking_output = [
                {"track_id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9}
            ]

            result = estimator.estimate_with_tracking(
                sample_frame, tracking_output, iou_threshold=0.5
            )

            assert len(result) == 1
            assert result[0]["track_id"] == 1
            assert result[0]["association_status"] == "matched"
        finally:
            patcher.stop()

    def test_with_no_matching_track(self, sample_frame):
        """Test pose estimation with no matching tracking output."""
        mock_result = make_single_person_result(bbox=(0, 0, 100, 100))
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model

            tracking_output = [
                {"track_id": 1, "bbox": [400, 400, 500, 500], "confidence": 0.9}
            ]

            result = estimator.estimate_with_tracking(
                sample_frame, tracking_output, iou_threshold=0.5
            )

            assert len(result) == 1
            assert result[0]["track_id"] is None
            assert result[0]["association_status"] == "unmatched"
        finally:
            patcher.stop()

    def test_with_empty_tracking(self, sample_frame):
        """Test with no tracking output."""
        mock_result = make_single_person_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model

            result = estimator.estimate_with_tracking(
                sample_frame, [], iou_threshold=0.5
            )

            assert len(result) == 1
            assert result[0]["track_id"] is None
            assert result[0]["association_status"] == "unmatched"
        finally:
            patcher.stop()

    def test_with_empty_poses(self, sample_frame):
        """Test with no poses detected."""
        mock_result = make_empty_result()
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            estimator = PoseEstimator(config={"verbose": False})
            estimator.model = mock_model

            result = estimator.estimate_with_tracking(
                sample_frame,
                [{"track_id": 1, "bbox": [0, 0, 100, 100]}],
                iou_threshold=0.5
            )

            assert result == []
        finally:
            patcher.stop()

    def test_uses_config_threshold_when_not_specified(self, sample_frame):
        """Test that estimate_with_tracking uses config association_iou_threshold."""
        mock_result = make_single_person_result(bbox=(100, 100, 200, 200))
        patcher, mock_model = patch_yolo_pose(mock_result)
        patcher.start()

        try:
            # Set a custom association threshold via config
            estimator = PoseEstimator(config={
                "verbose": False,
                "association_iou_threshold": 0.3
            })
            estimator.model = mock_model

            # Track with partial overlap (IoU < 0.5 but > 0.3)
            tracking_output = [
                {"track_id": 1, "bbox": [110, 110, 210, 210], "confidence": 0.9}
            ]

            # Without explicit iou_threshold, should use config (0.3)
            result = estimator.estimate_with_tracking(
                sample_frame, tracking_output
            )

            # IoU between [100,100,200,200] and [110,110,210,210] is ~0.33
            # With config threshold 0.3, this should match
            assert len(result) == 1
            assert result[0]["track_id"] == 1
            assert result[0]["association_status"] == "matched"
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Model smoke tests — require actual yolo11n-pose.pt
# ---------------------------------------------------------------------------

class TestPoseModelSmokeTest:
    """
    Smoke tests with the actual pretrained YOLO11n-pose model.

    These tests download and load the real model, so they may be slow.
    They are skipped if the model cannot be loaded (e.g., no internet).
    """

    @pytest.fixture
    def real_estimator(self):
        """Create PoseEstimator with actual model."""
        estimator = PoseEstimator(config={
            "device": "cpu",
            "verbose": False,
            "confidence_threshold": 0.5
        })
        success = estimator.load_model()
        if not success:
            pytest.skip("Could not load YOLO11n-pose model")
        return estimator

    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_model_loaded(self, real_estimator):
        """Test that the actual model loads."""
        assert real_estimator.model is not None
        info = real_estimator.get_model_info()
        assert info["is_loaded"] is True

    def test_model_info(self, real_estimator):
        """Test actual model info."""
        info = real_estimator.get_model_info()
        assert info["model_name"] == "yolo11n-pose.pt"
        assert info["device"] == "cpu"
        assert info["keypoint_count"] == 17

    def test_inference_on_noise(self, real_estimator, sample_frame):
        """Test inference on random noise frame (no persons expected at 0.5 threshold)."""
        poses = real_estimator.estimate(sample_frame)

        assert isinstance(poses, list)
        # Random noise with 0.5 confidence should not produce reliable detections
        assert len(poses) == 0

    def test_inference_returns_valid_structure(self, real_estimator, sample_frame):
        """Test that inference output structure is valid when poses exist."""
        # Create a more person-like frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:400, 200:500, :] = 200  # Person-like region
        # Add some texture
        for y in range(100, 400, 3):
            for x in range(200, 500, 3):
                frame[y, x] = np.random.randint(100, 255, 3)

        poses = real_estimator.estimate(frame)

        assert isinstance(poses, list)
        # Verify structure for any detected poses
        for pose in poses:
            assert "bbox" in pose
            assert "confidence" in pose
            assert "keypoints_xy" in pose
            assert "keypoints_conf" in pose
            assert "keypoints" in pose
            assert len(pose["bbox"]) == 4
            assert pose["keypoints_xy"].shape[1] == 17
            assert pose["keypoints_conf"].shape[0] == 17

    def test_performance_stats_after_inference(self, real_estimator, sample_frame):
        """Test performance stats after real inference."""
        real_estimator.reset_performance_stats()

        for _ in range(3):
            real_estimator.estimate(sample_frame)

        stats = real_estimator.get_performance_stats()
        assert stats["inference_count"] == 3
        assert stats["avg_latency_ms"] > 0
        assert stats["min_latency_ms"] > 0
