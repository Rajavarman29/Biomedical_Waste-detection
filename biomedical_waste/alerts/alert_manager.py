"""Threshold-based alerts and repeated high-risk streak handling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from biomedical_waste.config import (
    ALERT_COOLDOWN_SEC,
    HAZARD_ALERT_THRESHOLD,
    HIGH_RISK_STREAK_ALERT,
)
from biomedical_waste.hazard import PerDetectionHazard


@dataclass
class AlertEvent:
    timestamp: float
    message: str
    max_hazard: float
    streak: int
    level: str  # "warning" | "critical"


@dataclass
class AlertManager:
    hazard_threshold: float = HAZARD_ALERT_THRESHOLD
    streak_threshold: int = HIGH_RISK_STREAK_ALERT
    cooldown_sec: float = ALERT_COOLDOWN_SEC

    _streak: int = field(default=0, init=False)
    _last_alert_time: float = field(default=0.0, init=False)
    _events: List[AlertEvent] = field(default_factory=list, init=False)

    def reset_streak(self) -> None:
        self._streak = 0

    def evaluate(
        self,
        max_hazard: float,
        per_det: List[PerDetectionHazard],
        now: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        """
        Returns AlertEvent if threshold exceeded (respecting cooldown for repeated messages).
        Increments streak when hazard >= threshold; resets when below.
        """
        t = now if now is not None else time.time()
        high = max_hazard >= self.hazard_threshold

        if high:
            self._streak += 1
        else:
            self._streak = 0

        if not high:
            return None

        # Escalate if streak exceeds threshold
        level = "critical" if self._streak >= self.streak_threshold else "warning"
        msg = (
            f"Hazard {max_hazard:.2f} exceeds threshold {self.hazard_threshold:.2f}"
        )
        if self._streak >= self.streak_threshold:
            msg += f" | Sustained high risk ({self._streak} frames)"

        if t - self._last_alert_time < self.cooldown_sec and self._streak < self.streak_threshold:
            return None

        self._last_alert_time = t
        ev = AlertEvent(
            timestamp=t,
            message=msg,
            max_hazard=max_hazard,
            streak=self._streak,
            level=level,
        )
        self._events.append(ev)
        if len(self._events) > 500:
            self._events = self._events[-500:]
        return ev

    @property
    def recent_events(self) -> List[AlertEvent]:
        return list(self._events)
