"""Train YOLOv8 biomedical waste model: python -m biomedical_waste.scripts.train_cli"""

from __future__ import annotations

import argparse
from pathlib import Path

from biomedical_waste.config import (
    DEFAULT_DATA_YAML,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_MODEL,
    TRAIN_NAME,
)
from biomedical_waste.training import train_yolov8


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8n on biomedical waste (Roboflow YOLOv8 data.yaml)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    p.add_argument("--model", type=str, default=TRAIN_MODEL, help="Base weights e.g. yolov8n.pt")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    p.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    p.add_argument("--batch", type=int, default=TRAIN_BATCH)
    p.add_argument("--name", type=str, default=TRAIN_NAME)
    p.add_argument("--device", default=None, help="cuda:0, cpu, or mps")
    args = p.parse_args()

    train_yolov8(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )
    from biomedical_waste.config import PROJECT_ROOT

    best = PROJECT_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best}")


if __name__ == "__main__":
    main()
