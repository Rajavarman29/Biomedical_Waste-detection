"""Central configuration: class risks, paths, thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Final

# Project root (parent of biomedical_waste package)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Default paths (override via env or CLI)
DEFAULT_DATA_YAML: Final[Path] = PROJECT_ROOT / "data" / "data.yaml"
DEFAULT_WEIGHTS: Final[Path] = PROJECT_ROOT / "runs" / "detect" / "biomedical_waste_30ep" / "weights" / "best.pt"

# YOLO class names must match Roboflow export (order = class id)
# medical-waste-2 dataset uses this ordering (see data/medical-waste-2/data.yaml)
CLASS_NAMES: Final[tuple[str, ...]] = ("General", "Infectious", "Pathological", "Sharps")

# Base risk on 0–10 scale (spec: Sharps highest; Infectious/Pathological high; General low)
BASE_RISK: Final[Dict[str, float]] = {
    "Sharps": 10.0,
    "Infectious": 8.0,
    "Pathological": 8.0,
    "General": 3.0,
}

# Alert when normalized hazard score >= this (0–10)
HAZARD_ALERT_THRESHOLD: Final[float] = 6.5

# Repeated high-risk frames before escalating (webcam/video)
HIGH_RISK_STREAK_ALERT: Final[int] = 5

# Cooldown seconds between duplicate sound/log alerts (handled in alert manager)
ALERT_COOLDOWN_SEC: Final[float] = 2.0

# Inference
DEFAULT_IMGSZ: Final[int] = 640
DEFAULT_CONF: Final[float] = 0.25

# Training defaults (lightweight real-time model)
TRAIN_MODEL: Final[str] = "yolov8n.pt"
TRAIN_EPOCHS: Final[int] = 100
TRAIN_IMGSZ: Final[int] = 640
TRAIN_BATCH: Final[int] = 16
TRAIN_PROJECT: Final[str] = str(PROJECT_ROOT / "runs" / "detect")
TRAIN_NAME: Final[str] = "biomedical_waste"


def base_risk_for_class_id(class_id: int) -> float:
    if 0 <= class_id < len(CLASS_NAMES):
        return BASE_RISK.get(CLASS_NAMES[class_id], 5.0)
    return 5.0


def class_name_for_id(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return "Unknown"
