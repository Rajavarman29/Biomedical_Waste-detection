"""Train YOLOv8 (Ultralytics) on Roboflow-exported biomedical waste data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from biomedical_waste.config import (
    DEFAULT_DATA_YAML,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_MODEL,
    TRAIN_NAME,
    TRAIN_PROJECT,
)
from biomedical_waste.dataset import validate_yolo_dataset_yaml


def train_yolov8(
    data_yaml: Path | None = None,
    model: str = TRAIN_MODEL,
    epochs: int = TRAIN_EPOCHS,
    imgsz: int = TRAIN_IMGSZ,
    batch: int = TRAIN_BATCH,
    project: str = TRAIN_PROJECT,
    name: str = TRAIN_NAME,
    device: str | int | None = None,
    workers: int = 8,
    patience: int = 50,
    **kwargs: Any,
) -> Any:
    """
    Train a lightweight YOLOv8 model (default: yolov8n) and save weights under runs/detect/<name>/.
    Logs: TensorBoard-compatible results in runs/detect/<name>/.
    """
    from ultralytics import YOLO

    yaml_path = Path(data_yaml or DEFAULT_DATA_YAML).resolve()
    validate_yolo_dataset_yaml(yaml_path)

    model_obj = YOLO(model)
    results = model_obj.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        device=device,
        **kwargs,
    )
    return results
