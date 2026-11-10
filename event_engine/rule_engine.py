"""
Rule Engine — Phase 9
Evaluates configurable rules against per-frame pipeline data and emits events.

Rules are loaded from configs/alert_rules.yaml. Each rule specifies:
  - conditions (field, operator, value)
  - severity
  - cooldown_frames (duplicate suppression)

Supported condition fields
--------------------------
  occupancy_count   : int   — current tracked person count
  activity_label    : str   — any person in frame has this label
  anomaly_score     : float — max anomaly score across persons
  tracking_count    : int   — total active tracks
  zone_<id>_count   : int   — persons in zone with given id

Supported operators: gt, gte, lt, lte, eq, ne
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# path_utils lives in ai_engine/; event_engine is a sibling package
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from ai_engine.path_utils import get_config_path as _get_config_path

logger = logging.getLogger(__name__)


@dataclass
class RuleEvent:
    """
    Event produced when a rule condition is satisfied.

    Attributes
    ----------
    rule_id     : Unique rule identifier from alert_rules.yaml
    description : Human-readable rule description
    severity    : "INFO" | "WARNING" | "CRITICAL"
    frame_number: Frame at which the event was detected
    camera_id   : Camera source identifier
    evidence    : Dict of values that triggered the rule
    """
    rule_id:      str
    description:  str
    severity:     str
    frame_number: int
    camera_id:    str
    evidence:     Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """
    Evaluates alert rules against per-frame CV pipeline output.

    Instantiate once per pipeline; call ``evaluate()`` every frame.
    """

    _OPERATORS = {
        "gt":  lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "lt":  lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "eq":  lambda a, b: a == b,
        "ne":  lambda a, b: a != b,
    }

    def __init__(
        self,
        rules_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise RuleEngine.

        Args:
            rules_path : Path to alert_rules.yaml. Defaults to configs/alert_rules.yaml.
            config     : Optional dict of additional settings.
        """
        if rules_path is None:
            rules_path = str(_get_config_path("alert_rules.yaml"))
        self._rules: List[Dict[str, Any]] = []
        self._cooldowns: Dict[str, int] = {}  # rule_id -> last fired frame_number
        self._load_rules(rules_path)
        logger.info("RuleEngine initialised with %d rules", len(self._rules))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, frame_data: Dict[str, Any]) -> List[RuleEvent]:
        """
        Evaluate all rules against frame_data.

        Args:
            frame_data: Dictionary produced by the pipeline per frame, with keys:
                - frame_number    (int)
                - camera_id       (str)
                - tracking_output (List[Dict]) — from ByteTrackManager
                - occupancy_stats (Dict)        — from OccupancyManager
                - pose_results    (List[Dict])  — from ActivityRecognizer.update()
                - anomaly_results (List[Dict])  — from AnomalyDetector.score()

        Returns:
            List of RuleEvent objects for rules that fired this frame.
        """
        frame_number = frame_data.get("frame_number", 0)
        camera_id    = frame_data.get("camera_id", "unknown")

        # Build a flat feature dict for condition evaluation
        features = self._extract_features(frame_data)

        events: List[RuleEvent] = []
        for rule in self._rules:
            rule_id  = rule.get("id", "UNKNOWN")
            cooldown = rule.get("cooldown_frames", 0)

            # Cooldown check
            last_fired = self._cooldowns.get(rule_id, -cooldown - 1)
            if frame_number - last_fired <= cooldown:
                continue

            # Evaluate all conditions (all must be True — AND logic)
            conditions = rule.get("conditions", [])
            triggered, evidence = self._evaluate_conditions(conditions, features)

            if triggered:
                self._cooldowns[rule_id] = frame_number
                events.append(RuleEvent(
                    rule_id=rule_id,
                    description=rule.get("description", ""),
                    severity=rule.get("severity", "INFO"),
                    frame_number=frame_number,
                    camera_id=camera_id,
                    evidence=evidence,
                ))
                logger.debug("Rule fired: %s at frame %d", rule_id, frame_number)

        return events

    def get_rule_ids(self) -> List[str]:
        """Return list of loaded rule IDs."""
        return [r.get("id", "") for r in self._rules]

    def get_cooldown_state(self) -> Dict[str, int]:
        """Return copy of current cooldown state (for debugging)."""
        return dict(self._cooldowns)

    def reset_cooldowns(self) -> None:
        """Clear all cooldown state."""
        self._cooldowns.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rules(self, rules_path: str) -> None:
        """Load and validate rules from YAML file."""
        path = Path(rules_path)
        if not path.exists():
            logger.warning("alert_rules.yaml not found at %s — no rules loaded", rules_path)
            return
        try:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            self._rules = data.get("rules", []) if data else []
            # Validate required fields
            valid = []
            for rule in self._rules:
                if "id" not in rule or "conditions" not in rule:
                    logger.warning("Skipping invalid rule (missing id or conditions): %s", rule)
                    continue
                valid.append(rule)
            self._rules = valid
        except Exception as exc:
            logger.error("Failed to load alert rules: %s", exc)

    def _extract_features(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten per-frame pipeline data into a simple feature dict.

        Returns a dict suitable for condition evaluation.
        """
        features: Dict[str, Any] = {}

        # Occupancy
        occ = frame_data.get("occupancy_stats", {}).get("occupancy", {})
        features["occupancy_count"]    = occ.get("current_count", 0)
        features["occupancy_threshold"]= occ.get("alert_threshold", 0)

        # Tracking
        tracking = frame_data.get("tracking_output", [])
        features["tracking_count"] = len(tracking)

        # Activity labels — collect all unique labels present
        pose_results = frame_data.get("pose_results", [])
        activity_labels = {
            p.get("activity", {}).get("label", "UNKNOWN")
            for p in pose_results
        }
        # For condition evaluation: expose individual label presence
        features["_activity_labels"] = activity_labels
        # Also surface as a string field for eq/ne (checks membership)
        for label in activity_labels:
            features["activity_label"] = label   # will be overwritten — handled below

        # Anomaly scores — take maximum
        anomaly_results = frame_data.get("anomaly_results", [])
        if anomaly_results:
            features["anomaly_score"] = max(
                r.get("anomaly_score", 0.0) for r in anomaly_results
            )
        else:
            features["anomaly_score"] = 0.0

        return features

    def _evaluate_conditions(
        self,
        conditions: List[Dict[str, Any]],
        features: Dict[str, Any],
    ) -> tuple:
        """
        Evaluate all conditions (AND logic).

        Returns (all_satisfied: bool, evidence: Dict)
        """
        evidence: Dict[str, Any] = {}

        for cond in conditions:
            field_name = cond.get("field", "")
            operator   = cond.get("operator", "eq")
            value      = cond.get("value")

            op_fn = self._OPERATORS.get(operator)
            if op_fn is None:
                logger.warning("Unknown operator '%s' in rule condition", operator)
                return False, {}

            # Special handling for activity_label — checks membership
            if field_name == "activity_label":
                labels = features.get("_activity_labels", set())
                # For 'eq': at least one person has this label
                # For 'ne': no person has this label
                if operator == "eq":
                    satisfied = value in labels
                elif operator == "ne":
                    satisfied = value not in labels
                else:
                    satisfied = False
                field_value = list(labels)
            else:
                field_value = features.get(field_name, 0)
                try:
                    satisfied = op_fn(field_value, value)
                except (TypeError, ValueError):
                    satisfied = False

            evidence[field_name] = field_value

            if not satisfied:
                return False, {}

        return True, evidence
