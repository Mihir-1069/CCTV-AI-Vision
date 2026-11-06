"""
Event Engine Module
Provides rule-based event detection, alert generation, and event correlation.
"""

from .rule_engine import RuleEngine, RuleEvent
from .alert_generator import AlertGenerator, Alert
from .event_correlator import EventCorrelator, CorrelatedEventGroup

__all__ = [
    "RuleEngine",
    "RuleEvent",
    "AlertGenerator",
    "Alert",
    "EventCorrelator",
    "CorrelatedEventGroup",
]
