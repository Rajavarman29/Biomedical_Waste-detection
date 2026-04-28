"""Hazard score = base risk × confidence, normalized to 0–10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from biomedical_waste.config import base_risk_for_class_id
from biomedical_waste.detection import DetectionFrameResult


@dataclass
class PerDetectionHazard:
    class_id: int
    class_name: str
    confidence: float
    base_risk: float
    hazard_score: float  # 0–10


class HazardScoring:
    """
    Hazard Score = Base Risk (0–10) × Confidence → already in 0–10 scale
    (max when base_risk=10 and confidence=1 → 10).
    """

    @staticmethod
    def score_detection(class_id: int, confidence: float) -> float:
        br = base_risk_for_class_id(class_id)
        raw = br * float(confidence)
        return max(0.0, min(10.0, raw))

    def score_frame(self, det: DetectionFrameResult) -> List[PerDetectionHazard]:
        out: List[PerDetectionHazard] = []
        for i in range(det.num_detections):
            cid = det.class_ids[i]
            conf = det.confidences[i]
            name = det.class_names[i]
            br = base_risk_for_class_id(cid)
            hs = self.score_detection(cid, conf)
            out.append(
                PerDetectionHazard(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    base_risk=br,
                    hazard_score=hs,
                )
            )
        return out

    @staticmethod
    def frame_max_hazard(per_det: List[PerDetectionHazard]) -> float:
        if not per_det:
            return 0.0
        return max(p.hazard_score for p in per_det)
