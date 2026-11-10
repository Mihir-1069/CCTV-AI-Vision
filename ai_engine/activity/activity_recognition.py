"""
Activity Recognition Module
Provides rule-based human activity recognition from pose estimation output.

Classifies individual activities using:
- Joint angles from PoseAnalyzer (8 angles computed geometrically)
- Bounding box geometry (aspect ratio, vertical alignment)
- Temporal velocity computed from per-track bbox centre history
- Per-track temporal state with bounded memory (configurable deque)

Activity labels
---------------
STANDING        — upright posture, legs approximately straight
SITTING         — bent knees, compressed bounding box
WALKING         — upright posture, moderate normalised velocity
RUNNING         — upright posture, high normalised velocity (requires
                  sufficient temporal history)
FALL_LIKE_EVENT — sudden posture change to horizontal bounding box after
                  an observed upright history. This is a **geometric
                  observation only** and does NOT constitute a medical
                  diagnosis or confirmation of any medical event.
                  Requires human verification before any action is taken.
UNKNOWN         — insufficient evidence for any classification

Design decisions
----------------
- All geometry is 2D projection; angles may not reflect true 3D posture.
- Results degrade safely when keypoints are absent or low-confidence.
- Each track_id is processed independently; memory per track is bounded.
- WALKING and RUNNING require temporal history (track_id must be valid).
- FALL_LIKE_EVENT is never triggered for unmatched poses (track_id=None).
- Original pose dicts are never mutated; enriched copies are returned.
- Velocity is normalised to a reference frame width and reference FPS so
  that thresholds remain consistent across different camera resolutions
  and video frame rates.
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ai_engine.pose.pose_analyzer import PoseAnalyzer
from ai_engine.path_utils import get_config_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ActivityLabel(str, Enum):
    """Recognised activity labels produced by ActivityRecognizer."""
    STANDING        = "STANDING"
    SITTING         = "SITTING"
    WALKING         = "WALKING"
    RUNNING         = "RUNNING"
    FALL_LIKE_EVENT = "FALL_LIKE_EVENT"  # Geometric observation — not a medical diagnosis
    UNKNOWN         = "UNKNOWN"


@dataclass
class ActivityResult:
    """
    Result of activity classification for one person in one frame.

    Attributes
    ----------
    label        : ActivityLabel value
    confidence   : Score in [0.0, 1.0]
    evidence     : Human-readable list of conditions that produced the label
    frame_number : Frame index at which this result was computed
    track_id     : Anonymous temporary track ID, or None if unmatched
    """
    label: ActivityLabel
    confidence: float
    evidence: List[str]
    frame_number: int
    track_id: Optional[int]


# ---------------------------------------------------------------------------
# Internal temporal record
# ---------------------------------------------------------------------------

@dataclass
class _FrameRecord:
    """
    One entry in a track's temporal deque.

    Stores the per-frame label and bbox centre for velocity calculation.
    """
    frame_number: int
    label: ActivityLabel
    confidence: float
    bbox_center: Tuple[float, float]  # (cx, cy) in pixels


# ---------------------------------------------------------------------------
# ActivityRecognizer
# ---------------------------------------------------------------------------

class ActivityRecognizer:
    """
    Rule-based human activity recogniser.

    Consumes output from ``PoseEstimator.estimate_with_tracking()`` and
    returns the same list with an ``"activity"`` key added to each pose dict.

    Usage example
    -------------
    ::

        recognizer = ActivityRecognizer()

        # pose_results is the output of pose_estimator.estimate_with_tracking()
        enriched = recognizer.update(pose_results, frame_number=frame_idx,
                                     frame_width=640, fps=30.0)

        # Each element of enriched has:
        # enriched[i]["activity"] == {
        #     "label":      "STANDING",
        #     "confidence": 0.75,
        #     "evidence":   ["left_knee=178.0° > 155.0°", ...],
        # }

    Thread safety
    -------------
    Not thread-safe. Instantiate one recogniser per camera / pipeline.
    """

    # Defaults used when model.yaml has no ``activity:`` section
    _DEFAULTS: Dict[str, Any] = {
        # Minimum number of visible keypoints to attempt classification
        "min_visible_keypoints":      5,

        # Standing thresholds
        "standing_knee_angle_min":    155.0,   # degrees — legs nearly straight
        "standing_hip_angle_min":     150.0,   # degrees — torso upright
        "standing_aspect_ratio_min":    1.8,   # height/width
        "standing_vertical_min_px":    30.0,   # px shoulder–hip vertical gap

        # Sitting thresholds
        "sitting_knee_angle_max":     120.0,   # degrees — legs bent
        "sitting_hip_angle_max":      120.0,   # degrees — torso angled
        "sitting_aspect_ratio_max":     1.4,   # height/width

        # Walking / Running velocity (normalised px per frame at reference size)
        "walking_velocity_min":         2.0,
        "walking_velocity_max":        20.0,
        "walking_knee_angle_max":     165.0,
        "running_velocity_min":        15.0,
        "running_knee_angle_max":     150.0,
        "running_min_history_frames":   5,

        # Fall-like event thresholds
        "fall_aspect_ratio_max":        1.0,   # horizontal posture
        "fall_history_frames":         10,     # frames to look back for prior upright label
        "fall_min_visible_keypoints":   7,     # stricter than general min

        # Temporal state management
        "smoothing_window":            10,     # frames for majority vote
        "smoothing_confidence_min":     0.5,   # required majority fraction
        "fall_override_window":         5,     # frames for FALL_LIKE_EVENT override
        "track_max_gap":               30,     # frames before a track is evicted
        "track_history_len":           15,     # max deque entries per track

        # Reference values for velocity normalisation
        "reference_frame_width":      640,     # pixels
        "reference_fps":               30.0,   # frames per second
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Initialise ActivityRecognizer.

        Args:
            config      : Optional dict of overrides applied after loading
                          the YAML file (useful for tests).
            config_path : Path to YAML configuration file.
                          Defaults to ``configs/model.yaml`` relative to the
                          project root.
        """
        # 1. Start with built-in defaults
        self._cfg: Dict[str, Any] = dict(self._DEFAULTS)

        # 2. Resolve configuration file
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent / "configs" / "model.yaml"
            )

        # 3. Load from file (if present)
        if Path(config_path).exists():
            self._load_config(config_path)
        else:
            logger.warning(
                "Activity config file not found, using built-in defaults: %s",
                config_path,
            )

        # 4. Apply explicit overrides from caller
        if config:
            self._cfg.update(config)

        # 5. Internal PoseAnalyzer (threshold matches default; overriding via
        #    config is not needed because we pass raw arrays to analyze())
        self._analyzer = PoseAnalyzer(keypoint_threshold=0.5)

        # 6. Per-track temporal state: track_id -> deque[_FrameRecord]
        self._track_state: Dict[int, Deque[_FrameRecord]] = {}

        logger.info(
            "ActivityRecognizer initialised (window=%d, history_len=%d, max_gap=%d)",
            self._cfg["smoothing_window"],
            self._cfg["track_history_len"],
            self._cfg["track_max_gap"],
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load ``activity:`` section from YAML configuration file."""
        try:
            with open(config_path, "r") as fh:
                raw = yaml.safe_load(fh)
            activity_cfg = raw.get("activity", {}) if raw else {}
            self._cfg.update(activity_cfg)
            logger.info("Activity configuration loaded from %s", config_path)
        except Exception as exc:  # pragma: no cover — IO errors
            logger.error("Failed to load activity configuration: %s", exc)

    def get_config(self) -> Dict[str, Any]:
        """Return a copy of the current effective configuration."""
        return dict(self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        pose_results: List[Dict[str, Any]],
        frame_number: int,
        frame_width: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify activity for each person in *pose_results*.

        Returns a **new list** of pose dicts. Each dict is a shallow copy of
        the input dict with an ``"activity"`` key added::

            pose["activity"] = {
                "label":      str,         # ActivityLabel value
                "confidence": float,       # 0.0 – 1.0
                "evidence":   List[str],   # human-readable reasoning
            }

        Original pose dicts are **not mutated**.

        Args:
            pose_results : Output of ``PoseEstimator.estimate_with_tracking()``.
            frame_number : Current frame index (monotonically increasing).
            frame_width  : Actual frame width in pixels used for velocity
                           normalisation. Falls back to config
                           ``reference_frame_width`` if None.
            fps          : Actual video FPS used for velocity normalisation.
                           Falls back to config ``reference_fps`` if None.

        Returns:
            New list of enriched pose dicts.
        """
        if not pose_results:
            self._evict_stale_tracks(frame_number)
            return []

        # Velocity normalisation factors
        effective_width = frame_width if frame_width is not None else self._cfg["reference_frame_width"]
        effective_fps   = fps         if fps         is not None else self._cfg["reference_fps"]

        # Scale raw pixel displacement per frame → reference-resolution-equivalent
        px_scale  = self._cfg["reference_frame_width"] / max(effective_width, 1)
        fps_scale = self._cfg["reference_fps"] / max(effective_fps, 1.0)
        velocity_scale = px_scale * fps_scale

        enriched: List[Dict[str, Any]] = []

        for pose in pose_results:
            track_id     = pose.get("track_id")          # int or None
            kp_xy        = pose.get("keypoints_xy")
            kp_conf      = pose.get("keypoints_conf")
            bbox         = pose.get("bbox", [])
            assoc_status = pose.get("association_status", "unmatched")

            # ---- Single-frame geometric classification ----
            if kp_xy is None or kp_conf is None:
                frame_label    = ActivityLabel.UNKNOWN
                frame_conf     = 0.0
                frame_evidence = ["keypoints_xy or keypoints_conf missing from pose dict"]
            else:
                frame_label, frame_conf, frame_evidence = self._classify_single_frame(
                    kp_xy=kp_xy,
                    kp_conf=kp_conf,
                    bbox=bbox,
                    track_id=track_id,
                    frame_number=frame_number,
                    velocity_scale=velocity_scale,
                    assoc_status=assoc_status,
                )

            # ---- Temporal state update + smoothing ----
            has_valid_track = (track_id is not None and track_id >= 0)

            if has_valid_track:
                bbox_center = self._bbox_center(bbox)
                self._push_track_state(
                    track_id, frame_number, frame_label, frame_conf, bbox_center
                )
                final_label, final_conf = self._smooth_label(track_id, frame_number)
            else:
                # Unmatched / uncertain pose — no temporal state available.
                # WALKING, RUNNING, FALL_LIKE_EVENT require temporal evidence.
                if frame_label in (
                    ActivityLabel.WALKING,
                    ActivityLabel.RUNNING,
                    ActivityLabel.FALL_LIKE_EVENT,
                ):
                    final_label = ActivityLabel.UNKNOWN
                    final_conf  = 0.0
                    frame_evidence = frame_evidence + [
                        "temporal tracking required for this label; track_id is None"
                    ]
                else:
                    final_label = frame_label
                    final_conf  = frame_conf

            # ---- Build enriched copy (never mutate original) ----
            # UNKNOWN confidence is semantically undefined — always report 0.0
            if final_label == ActivityLabel.UNKNOWN:
                final_conf = 0.0

            enriched_pose = dict(pose)
            enriched_pose["activity"] = {
                "label":      final_label.value,
                "confidence": round(float(final_conf), 4),
                "evidence":   list(frame_evidence),
            }
            enriched.append(enriched_pose)

        # Evict tracks absent for more than track_max_gap frames
        self._evict_stale_tracks(frame_number)
        return enriched

    def reset(self) -> None:
        """Clear all per-track temporal state."""
        self._track_state.clear()
        logger.info("ActivityRecognizer state reset")

    def get_track_ids(self) -> List[int]:
        """Return list of track IDs currently held in state."""
        return list(self._track_state.keys())

    def get_track_history(self, track_id: int) -> List[Dict[str, Any]]:
        """
        Return the temporal history for *track_id* as a list of plain dicts.

        Useful for debugging and testing. Returns ``[]`` for unknown tracks.
        """
        if track_id not in self._track_state:
            return []
        return [
            {
                "frame_number": r.frame_number,
                "label":        r.label.value,
                "confidence":   r.confidence,
                "bbox_center":  r.bbox_center,
            }
            for r in self._track_state[track_id]
        ]

    # ------------------------------------------------------------------
    # Single-frame classification
    # ------------------------------------------------------------------

    def _classify_single_frame(
        self,
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
        bbox: List[int],
        track_id: Optional[int],
        frame_number: int,
        velocity_scale: float,
        assoc_status: str,
    ) -> Tuple[ActivityLabel, float, List[str]]:
        """
        Classify one person in one frame using geometric rules only.

        Calls ``PoseAnalyzer.analyze()`` internally to obtain joint angles.

        Returns:
            (label, confidence, evidence_list)
        """
        analysis  = self._analyzer.analyze(kp_xy, kp_conf)
        angles    = analysis["joint_angles"]
        vis_stats = analysis["visibility_stats"]
        visible_kp = vis_stats["visible_keypoints"]

        # Minimum keypoint gate
        min_kp = self._cfg["min_visible_keypoints"]
        if visible_kp < min_kp:
            return (
                ActivityLabel.UNKNOWN,
                0.0,
                [f"only {visible_kp}/{17} keypoints visible (need {min_kp})"],
            )

        aspect_ratio   = self._bbox_aspect_ratio(bbox)
        geometry_ev    = [f"aspect_ratio={aspect_ratio:.2f}, visible_kp={visible_kp}"]

        has_valid_track = (track_id is not None and track_id >= 0)

        # ---- FALL_LIKE_EVENT (tracked persons only) ----
        # Three-condition check; conservative and never fires for track_id=None
        if has_valid_track:
            fall_label, fall_conf, fall_ev = self._check_fall_like(
                aspect_ratio=aspect_ratio,
                track_id=track_id,
                frame_number=frame_number,
                kp_conf=kp_conf,
            )
            if fall_label == ActivityLabel.FALL_LIKE_EVENT:
                return fall_label, fall_conf, fall_ev

        # ---- SITTING ----
        sit_score, sit_ev = self._score_sitting(angles, aspect_ratio)
        if sit_score >= 2:
            conf = min(1.0, sit_score / 5.0)
            return ActivityLabel.SITTING, conf, sit_ev + geometry_ev

        # ---- STANDING (and velocity-based WALKING / RUNNING) ----
        stand_score, stand_ev = self._score_standing(angles, aspect_ratio, kp_xy, kp_conf)
        if stand_score >= 2:
            stand_conf = min(1.0, stand_score / 6.0)

            # For tracked persons try velocity-based upgrade
            if has_valid_track:
                motion_label, motion_conf, motion_ev = self._check_velocity_activity(
                    track_id=track_id,
                    frame_number=frame_number,
                    angles=angles,
                    velocity_scale=velocity_scale,
                )
                if motion_label in (ActivityLabel.WALKING, ActivityLabel.RUNNING):
                    return motion_label, motion_conf, motion_ev + stand_ev

            return ActivityLabel.STANDING, stand_conf, stand_ev + geometry_ev

        return ActivityLabel.UNKNOWN, 0.0, ["insufficient geometric evidence"] + geometry_ev

    # ------------------------------------------------------------------
    # Scoring helpers — return (score, evidence_list)
    # ------------------------------------------------------------------

    def _score_standing(
        self,
        angles: Dict[str, Optional[float]],
        aspect_ratio: float,
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
    ) -> Tuple[int, List[str]]:
        """
        Score STANDING evidence.  Each satisfied condition adds 1 to the score.
        Score ≥ 2 is required for a STANDING classification.
        """
        cfg    = self._cfg
        score  = 0
        ev: List[str] = []

        # 1 & 2 — Knee angles (legs approximately straight)
        lk = angles.get("left_knee")
        rk = angles.get("right_knee")
        thr_knee = cfg["standing_knee_angle_min"]
        if lk is not None and lk > thr_knee:
            score += 1
            ev.append(f"left_knee={lk:.1f}° > {thr_knee}°")
        if rk is not None and rk > thr_knee:
            score += 1
            ev.append(f"right_knee={rk:.1f}° > {thr_knee}°")

        # 3 & 4 — Hip angles (torso upright)
        lh = angles.get("left_hip")
        rh = angles.get("right_hip")
        thr_hip = cfg["standing_hip_angle_min"]
        if lh is not None and lh > thr_hip:
            score += 1
            ev.append(f"left_hip={lh:.1f}° > {thr_hip}°")
        if rh is not None and rh > thr_hip:
            score += 1
            ev.append(f"right_hip={rh:.1f}° > {thr_hip}°")

        # 5 — Aspect ratio (tall bounding box)
        thr_ar = cfg["standing_aspect_ratio_min"]
        if aspect_ratio > thr_ar:
            score += 1
            ev.append(f"aspect_ratio={aspect_ratio:.2f} > {thr_ar}")

        # 6 — Vertical shoulder–hip separation (shoulders above hips)
        vert = self._shoulder_hip_vertical(kp_xy, kp_conf)
        thr_vert = cfg["standing_vertical_min_px"]
        if vert is not None and vert > thr_vert:
            score += 1
            ev.append(f"shoulder_hip_vertical={vert:.1f}px > {thr_vert}px")

        return score, ev

    def _score_sitting(
        self,
        angles: Dict[str, Optional[float]],
        aspect_ratio: float,
    ) -> Tuple[int, List[str]]:
        """
        Score SITTING evidence.  Score ≥ 2 is required for classification.
        """
        cfg    = self._cfg
        score  = 0
        ev: List[str] = []

        # 1 & 2 — Knee angles (legs bent)
        lk = angles.get("left_knee")
        rk = angles.get("right_knee")
        thr_knee = cfg["sitting_knee_angle_max"]
        if lk is not None and lk < thr_knee:
            score += 1
            ev.append(f"left_knee={lk:.1f}° < {thr_knee}°")
        if rk is not None and rk < thr_knee:
            score += 1
            ev.append(f"right_knee={rk:.1f}° < {thr_knee}°")

        # 3 & 4 — Hip angles (torso angled forward)
        lh = angles.get("left_hip")
        rh = angles.get("right_hip")
        thr_hip = cfg["sitting_hip_angle_max"]
        if lh is not None and lh < thr_hip:
            score += 1
            ev.append(f"left_hip={lh:.1f}° < {thr_hip}°")
        if rh is not None and rh < thr_hip:
            score += 1
            ev.append(f"right_hip={rh:.1f}° < {thr_hip}°")

        # 5 — Aspect ratio (compressed bounding box)
        thr_ar = cfg["sitting_aspect_ratio_max"]
        if aspect_ratio < thr_ar:
            score += 1
            ev.append(f"aspect_ratio={aspect_ratio:.2f} < {thr_ar}")

        return score, ev

    def _check_velocity_activity(
        self,
        track_id: int,
        frame_number: int,
        angles: Dict[str, Optional[float]],
        velocity_scale: float,
    ) -> Tuple[ActivityLabel, float, List[str]]:
        """
        Determine WALKING or RUNNING from normalised velocity history.

        Returns ``(UNKNOWN, 0.0, [...])`` when insufficient data or velocity
        is outside the configured ranges.
        """
        cfg     = self._cfg
        history = self._track_state.get(track_id)

        if not history or len(history) < 2:
            return ActivityLabel.UNKNOWN, 0.0, ["insufficient velocity history (need ≥2 frames)"]

        # Compute per-interval normalised velocities
        records    = list(history)
        velocities: List[float] = []
        for i in range(1, len(records)):
            prev      = records[i - 1]
            curr      = records[i]
            frame_gap = curr.frame_number - prev.frame_number
            if frame_gap <= 0:
                continue
            dx   = curr.bbox_center[0] - prev.bbox_center[0]
            dy   = curr.bbox_center[1] - prev.bbox_center[1]
            dist = float(np.sqrt(dx**2 + dy**2))
            # Normalise: scale to reference resolution and per-frame rate
            norm_vel = (dist / frame_gap) * velocity_scale
            velocities.append(norm_vel)

        if not velocities:
            return ActivityLabel.UNKNOWN, 0.0, ["no velocity computable from history"]

        avg_vel = float(np.mean(velocities))
        ev_base = [
            f"avg_velocity={avg_vel:.3f} norm_px/frame "
            f"(scale={velocity_scale:.4f})"
        ]

        # --- RUNNING ---
        min_hist  = cfg["running_min_history_frames"]
        run_v_min = cfg["running_velocity_min"]
        run_k_max = cfg["running_knee_angle_max"]

        lk = angles.get("left_knee")
        rk = angles.get("right_knee")
        knee_bent_run = (
            (lk is not None and lk < run_k_max)
            or (rk is not None and rk < run_k_max)
        )

        if avg_vel >= run_v_min and knee_bent_run and len(records) >= min_hist:
            conf = min(1.0, avg_vel / (run_v_min * 2.0))
            return ActivityLabel.RUNNING, conf, ev_base + [
                f"velocity={avg_vel:.2f} >= running_min={run_v_min}",
                f"knee angle < {run_k_max}°",
                f"history={len(records)} frames >= {min_hist}",
            ]

        # --- WALKING ---
        walk_v_min = cfg["walking_velocity_min"]
        walk_v_max = cfg["walking_velocity_max"]

        if walk_v_min <= avg_vel <= walk_v_max:
            conf = min(1.0, avg_vel / walk_v_max)
            return ActivityLabel.WALKING, conf, ev_base + [
                f"velocity in [{walk_v_min}, {walk_v_max}] (walking range)"
            ]

        return ActivityLabel.UNKNOWN, 0.0, ev_base + [
            "velocity outside walking/running ranges"
        ]

    def _check_fall_like(
        self,
        aspect_ratio: float,
        track_id: int,
        frame_number: int,
        kp_conf: np.ndarray,
    ) -> Tuple[ActivityLabel, float, List[str]]:
        """
        Conservative FALL_LIKE_EVENT check.  All three conditions must hold:

        1. Current bounding box aspect ratio < ``fall_aspect_ratio_max``
           (person appears horizontal / compressed).
        2. The track had an **upright** label (STANDING, WALKING, or RUNNING)
           within the last ``fall_history_frames`` frames.
        3. Keypoint evidence is sufficient (at least
           ``fall_min_visible_keypoints`` keypoints above threshold), so that
           poor pose quality does not spuriously trigger this label.

        Returns ``(FALL_LIKE_EVENT, confidence, evidence)`` when all three
        conditions are satisfied, otherwise ``(UNKNOWN, 0.0, [])``.

        This is a **geometric observation only**.  It does not diagnose,
        confirm, or assess the severity of any medical event.
        """
        cfg = self._cfg

        # Condition 1 — horizontal bounding box
        if aspect_ratio >= cfg["fall_aspect_ratio_max"]:
            return ActivityLabel.UNKNOWN, 0.0, []

        # Condition 3 — keypoint quality
        visible_kp = int(np.sum(np.array(kp_conf, dtype=np.float64) >= 0.5))
        min_kp = cfg["fall_min_visible_keypoints"]
        if visible_kp < min_kp:
            return ActivityLabel.UNKNOWN, 0.0, []

        # Condition 2 — prior upright history
        history = self._track_state.get(track_id)
        if not history:
            return ActivityLabel.UNKNOWN, 0.0, []

        lookback       = cfg["fall_history_frames"]
        upright_labels = {ActivityLabel.STANDING, ActivityLabel.WALKING, ActivityLabel.RUNNING}
        recent_upright = any(
            r.label in upright_labels
            and (frame_number - r.frame_number) <= lookback
            for r in history
        )

        if not recent_upright:
            return ActivityLabel.UNKNOWN, 0.0, []

        # All three conditions met
        evidence = [
            f"aspect_ratio={aspect_ratio:.2f} < {cfg['fall_aspect_ratio_max']} (horizontal posture)",
            f"upright label detected within last {lookback} frames",
            f"visible keypoints={visible_kp} >= {min_kp} (quality threshold)",
            "FALL_LIKE_EVENT: geometric observation only — human verification required",
        ]
        return ActivityLabel.FALL_LIKE_EVENT, 0.9, evidence

    # ------------------------------------------------------------------
    # Temporal state management
    # ------------------------------------------------------------------

    def _push_track_state(
        self,
        track_id: int,
        frame_number: int,
        label: ActivityLabel,
        confidence: float,
        bbox_center: Tuple[float, float],
    ) -> None:
        """Append a frame record to the track's bounded deque."""
        max_len = int(self._cfg["track_history_len"])
        if track_id not in self._track_state:
            self._track_state[track_id] = deque(maxlen=max_len)
        self._track_state[track_id].append(
            _FrameRecord(
                frame_number=frame_number,
                label=label,
                confidence=confidence,
                bbox_center=bbox_center,
            )
        )

    def _smooth_label(
        self,
        track_id: int,
        frame_number: int,
    ) -> Tuple[ActivityLabel, float]:
        """
        Apply majority-vote smoothing over the recent temporal window.

        Special rule: FALL_LIKE_EVENT overrides majority voting.
        If any record within the last ``fall_override_window`` frames carries
        FALL_LIKE_EVENT, it is immediately returned (high-urgency signal).
        """
        cfg     = self._cfg
        history = self._track_state.get(track_id)
        if not history:
            return ActivityLabel.UNKNOWN, 0.0

        # FALL_LIKE_EVENT override — check the short fall window first
        fall_window = int(cfg["fall_override_window"])
        for record in history:
            if (
                record.label == ActivityLabel.FALL_LIKE_EVENT
                and (frame_number - record.frame_number) <= fall_window
            ):
                fall_count = sum(
                    1 for r in history
                    if r.label == ActivityLabel.FALL_LIKE_EVENT
                    and (frame_number - r.frame_number) <= fall_window
                )
                total_in_window = sum(
                    1 for r in history
                    if (frame_number - r.frame_number) <= fall_window
                )
                conf = fall_count / max(total_in_window, 1)
                return ActivityLabel.FALL_LIKE_EVENT, min(1.0, conf)

        # Majority vote over the main smoothing window
        window = int(cfg["smoothing_window"])
        window_labels: List[ActivityLabel] = [
            r.label
            for r in history
            if (frame_number - r.frame_number) <= window
        ]

        if not window_labels:
            return ActivityLabel.UNKNOWN, 0.0

        counts                       = Counter(window_labels)
        majority_label, majority_cnt = counts.most_common(1)[0]
        majority_ratio               = majority_cnt / len(window_labels)

        min_conf = float(cfg["smoothing_confidence_min"])
        if majority_ratio >= min_conf:
            return majority_label, float(majority_ratio)

        return ActivityLabel.UNKNOWN, 0.0

    def _evict_stale_tracks(self, frame_number: int) -> None:
        """Remove tracks whose last record is older than ``track_max_gap`` frames."""
        max_gap   = int(self._cfg["track_max_gap"])
        to_remove = [
            tid
            for tid, hist in self._track_state.items()
            if not hist or (frame_number - hist[-1].frame_number) > max_gap
        ]
        for tid in to_remove:
            del self._track_state[tid]
            logger.debug("Evicted stale track %d at frame %d", tid, frame_number)

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_aspect_ratio(bbox: List[int]) -> float:
        """
        Height / Width of bounding box.

        Returns 0.0 for invalid or zero-width bounding boxes.
        """
        if len(bbox) < 4:
            return 0.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0:
            return 0.0
        return float(h) / float(w)

    @staticmethod
    def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
        """Centre (cx, cy) of bounding box."""
        if len(bbox) < 4:
            return (0.0, 0.0)
        return (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )

    def _shoulder_hip_vertical(
        self,
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
        threshold: float = 0.5,
    ) -> Optional[float]:
        """
        Vertical pixel gap between shoulder midpoint and hip midpoint.

        Returns a positive value when shoulders are above hips (upright
        posture), negative when inverted, or ``None`` if required keypoints
        are unavailable.

        Indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12.
        In image coordinates y increases downward, so hip_y > shoulder_y for
        an upright person → positive return value.
        """
        shoulder_ys: List[float] = []
        if float(kp_conf[5]) >= threshold:
            shoulder_ys.append(float(kp_xy[5][1]))
        if float(kp_conf[6]) >= threshold:
            shoulder_ys.append(float(kp_xy[6][1]))

        hip_ys: List[float] = []
        if float(kp_conf[11]) >= threshold:
            hip_ys.append(float(kp_xy[11][1]))
        if float(kp_conf[12]) >= threshold:
            hip_ys.append(float(kp_xy[12][1]))

        if not shoulder_ys or not hip_ys:
            return None

        return float(np.mean(hip_ys)) - float(np.mean(shoulder_ys))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ActivityRecognizer("
            f"active_tracks={len(self._track_state)}, "
            f"window={self._cfg['smoothing_window']})"
        )
