"""YOLOv8 multi-object detection for images, webcam, and video streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from biomedical_waste.config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_WEIGHTS,
    class_name_for_id,
)


@dataclass
class DetectionFrameResult:
    """Single frame inference output."""

    frame_bgr: np.ndarray
    boxes_xyxy: np.ndarray  # (N, 4) float32
    class_ids: List[int]
    confidences: List[float]
    class_names: List[str]
    inference_ms: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.class_ids)


class BiomedicalDetector:
    """
    Real-time biomedical waste detector (Ultralytics YOLOv8).
    Optimized for CPU/GPU; uses imgsz and half=False on CPU for stability.
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | int | None = None,
    ) -> None:
        from ultralytics import YOLO

        w = Path(weights or DEFAULT_WEIGHTS)
        if not w.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {w}\n"
                "Train first: python -m biomedical_waste.scripts.train_cli"
            )
        self.model = YOLO(str(w))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def _predict_numpy(self, frame_bgr: np.ndarray) -> Tuple[Any, float]:
        import time

        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results[0], elapsed_ms

    def _result_to_frame(
        self, frame_bgr: np.ndarray, result: Any, inference_ms: float
    ) -> DetectionFrameResult:
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionFrameResult(
                frame_bgr=frame_bgr,
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                class_ids=[],
                confidences=[],
                class_names=[],
                inference_ms=inference_ms,
            )
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int).tolist()
        conf = result.boxes.conf.cpu().numpy().tolist()
        names = [class_name_for_id(int(c)) for c in cls]
        return DetectionFrameResult(
            frame_bgr=frame_bgr,
            boxes_xyxy=xyxy,
            class_ids=cls,
            confidences=conf,
            class_names=names,
            inference_ms=inference_ms,
        )

    def predict_image(self, image_path: Union[str, Path]) -> DetectionFrameResult:
        """Load image from disk and run detection."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        result, ms = self._predict_numpy(bgr)
        return self._result_to_frame(bgr, result, ms)

    def predict_frame(self, frame_bgr: np.ndarray) -> DetectionFrameResult:
        """Detect on a single BGR frame (OpenCV convention)."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be HxWx3 BGR uint8")
        result, ms = self._predict_numpy(frame_bgr)
        return self._result_to_frame(frame_bgr.copy(), result, ms)

    def annotate_frame(
        self,
        det: DetectionFrameResult,
        line_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw boxes and labels on a copy of the frame."""
        out = det.frame_bgr.copy()
        for i in range(det.num_detections):
            x1, y1, x2, y2 = map(int, det.boxes_xyxy[i])
            label = f"{det.class_names[i]} {det.confidences[i]:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), line_thickness)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return out

    def iter_webcam(
        self,
        camera_id: int = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[DetectionFrameResult, None, None]:
        """Real-time webcam stream."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {camera_id}")
        try:
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.predict_frame(frame)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            cap.release()

    def iter_video(
        self,
        source: Union[str, Path, int],
        max_frames: Optional[int] = None,
    ) -> Generator[DetectionFrameResult, None, None]:
        """Video file path or camera index."""
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        try:
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.predict_frame(frame)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            cap.release()
