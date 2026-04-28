"""Disposal recommendations from aggregated hazard score."""

from __future__ import annotations

from enum import Enum
from typing import Tuple


class DisposalTier(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


class DisposalEngine:
    """
    Maps max frame hazard (0–10) to action tier:
    - Critical → Immediate isolation
    - High → Priority disposal
    - Moderate → Regulated handling
    - Low → Standard disposal
    """

    @staticmethod
    def tier_from_score(max_hazard: float) -> DisposalTier:
        if max_hazard >= 8.0:
            return DisposalTier.CRITICAL
        if max_hazard >= 6.0:
            return DisposalTier.HIGH
        if max_hazard >= 3.5:
            return DisposalTier.MODERATE
        return DisposalTier.LOW

    @staticmethod
    def recommend(max_hazard: float) -> Tuple[DisposalTier, str]:
        tier = DisposalEngine.tier_from_score(max_hazard)
        text = {
            DisposalTier.CRITICAL: "Immediate isolation: secure area, use PPE, notify safety officer.",
            DisposalTier.HIGH: "Priority disposal: labeled sharps/biohazard containers, expedited pickup.",
            DisposalTier.MODERATE: "Regulated handling: segregate per protocol, documented chain of custody.",
            DisposalTier.LOW: "Standard disposal: follow facility general waste guidelines.",
        }[tier]
        return tier, text
