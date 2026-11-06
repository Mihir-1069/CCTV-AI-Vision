"""
Alert Generator — Phase 10
Converts RuleEngine events into structured, deduplicated Alert objects.

Design
------
- Each Alert has a unique ID, severity, timestamp, event type, and evidence.
- Duplicate suppression: same rule_id + same camera_id suppressed within
  a configurable cooldown window (seconds-based, not frame-based).
- Alerts are JSON-serialisable for downstream consumption.
- External notification integrations (email, webhook, SMS) are out of scope.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from event_engine.rule_engine import RuleEvent

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """
    Structured alert produced by the AlertGenerator.

    Attributes
    ----------
    alert_id    : Unique UUID string for this alert
    rule_id     : Rule that triggered the alert
    severity    : "INFO" | "WARNING" | "CRITICAL"
    timestamp   : ISO 8601 UTC timestamp
    camera_id   : Camera source
    frame_number: Frame number at detection time
    description : Human-readable description
    evidence    : Dict of values that triggered the rule
    """
    alert_id:     str
    rule_id:      str
    severity:     str
    timestamp:    str
    camera_id:    str
    frame_number: int
    description:  str
    evidence:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict representation."""
        return asdict(self)

    def is_critical(self) -> bool:
        """True if severity is CRITICAL."""
        return self.severity == "CRITICAL"


class AlertGenerator:
    """
    Converts RuleEngine RuleEvents into deduplicated Alert objects.

    Instantiate once per pipeline; call ``generate()`` for each event.
    Duplicates within ``dedup_window_seconds`` are suppressed.
    """

    _DEFAULT_DEDUP_WINDOW = 10.0   # seconds

    def __init__(
        self,
        dedup_window_seconds: float = _DEFAULT_DEDUP_WINDOW,
        max_history: int = 500,
    ) -> None:
        """
        Initialise AlertGenerator.

        Args:
            dedup_window_seconds: Suppress identical (rule_id, camera_id) alerts
                                  within this time window.
            max_history          : Maximum number of alerts to retain in history.
        """
        self._dedup_window   = dedup_window_seconds
        self._max_history    = max_history
        self._last_generated: Dict[str, float] = {}   # dedup_key -> wall time
        self._alert_history:  List[Alert] = []

        logger.info("AlertGenerator initialised (dedup_window=%.1fs)", dedup_window_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, event: RuleEvent) -> Optional[Alert]:
        """
        Generate an Alert from a RuleEvent, with duplicate suppression.

        Args:
            event : RuleEvent from RuleEngine.evaluate()

        Returns:
            Alert object if not suppressed, None if duplicate.
        """
        dedup_key = f"{event.camera_id}::{event.rule_id}"
        now = time.monotonic()

        last_time = self._last_generated.get(dedup_key, -self._dedup_window - 1)
        if now - last_time < self._dedup_window:
            logger.debug("Alert suppressed (dedup): %s", dedup_key)
            return None

        self._last_generated[dedup_key] = now

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=event.rule_id,
            severity=event.severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=event.camera_id,
            frame_number=event.frame_number,
            description=event.description,
            evidence=dict(event.evidence),
        )

        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        log_level = logging.CRITICAL if alert.is_critical() else logging.WARNING
        logger.log(log_level, "[ALERT] %s | %s | cam=%s | frame=%d",
                   alert.severity, alert.rule_id, alert.camera_id, alert.frame_number)

        return alert

    def generate_batch(self, events: List[RuleEvent]) -> List[Alert]:
        """
        Generate alerts for a list of events.

        Returns:
            List of non-suppressed Alert objects.
        """
        alerts = []
        for event in events:
            alert = self.generate(event)
            if alert:
                alerts.append(alert)
        return alerts

    def get_history(self, limit: Optional[int] = None) -> List[Alert]:
        """
        Return alert history.

        Args:
            limit : Max entries to return (None = all).
        """
        if limit is None:
            return list(self._alert_history)
        return list(self._alert_history[-limit:])

    def get_critical_count(self) -> int:
        """Return count of CRITICAL alerts in history."""
        return sum(1 for a in self._alert_history if a.is_critical())

    def clear_history(self) -> None:
        """Clear alert history and dedup state."""
        self._alert_history.clear()
        self._last_generated.clear()

    def reset_dedup(self) -> None:
        """Clear only the dedup state (keep history)."""
        self._last_generated.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics over the alert history."""
        severity_counts: Dict[str, int] = {}
        rule_counts:     Dict[str, int] = {}
        for a in self._alert_history:
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
            rule_counts[a.rule_id]      = rule_counts.get(a.rule_id, 0) + 1
        return {
            "total_alerts":     len(self._alert_history),
            "by_severity":      severity_counts,
            "by_rule":          rule_counts,
            "dedup_window_sec": self._dedup_window,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AlertGenerator("
            f"history={len(self._alert_history)}, "
            f"dedup_window={self._dedup_window}s)"
        )
