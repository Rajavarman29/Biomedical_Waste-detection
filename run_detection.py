"""
CLI: biomedical waste detection on image, webcam, or video file.
Computes hazard scores, disposal tier, and console alerts.

Usage (from project root):
  python run_detection.py --source image.jpg
  python run_detection.py --webcam
  python run_detection.py --video clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from biomedical_waste.alerts import AlertManager
from biomedical_waste.config import DEFAULT_WEIGHTS, HAZARD_ALERT_THRESHOLD
from biomedical_waste.detection import BiomedicalDetector
from biomedical_waste.disposal import DisposalEngine
from biomedical_waste.hazard import HazardScoring


def run_image(path: Path, weights: Path, conf: float, show: bool) -> None:
    det = BiomedicalDetector(weights=weights, conf=conf)
    hs = HazardScoring()
    disp = DisposalEngine()
    alerts = AlertManager(hazard_threshold=HAZARD_ALERT_THRESHOLD)

    r = det.predict_image(path)
    per = hs.score_frame(r)
    max_h = hs.frame_max_hazard(per)
    tier, rec = disp.recommend(max_h)
    ev = alerts.evaluate(max_h, per)

    print(f"Detections: {r.num_detections} | max hazard: {max_h:.2f} | tier: {tier.value}")
    print(f"Disposal: {rec}")
    if ev:
        print(f"ALERT [{ev.level}]: {ev.message}")

    for p in per:
        print(f"  - {p.class_name}: conf={p.confidence:.3f} hazard={p.hazard_score:.2f}")

    if show:
        vis = det.annotate_frame(r)
        cv2.imshow("Biomedical Waste Detection", vis)
        print("Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(
    camera_id: int,
    weights: Path,
    conf: float,
    window: bool,
    max_frames: int | None,
) -> None:
    det = BiomedicalDetector(weights=weights, conf=conf)
    hs = HazardScoring()
    disp = DisposalEngine()
    alerts = AlertManager(hazard_threshold=HAZARD_ALERT_THRESHOLD)

    for frame_result in det.iter_webcam(camera_id=camera_id, max_frames=max_frames):
        if not _process_frame(det, hs, disp, alerts, frame_result, window):
            break
    _finish_stream(window)


def run_video(
    video_path: Path,
    weights: Path,
    conf: float,
    window: bool,
    max_frames: int | None,
) -> None:
    det = BiomedicalDetector(weights=weights, conf=conf)
    hs = HazardScoring()
    disp = DisposalEngine()
    alerts = AlertManager(hazard_threshold=HAZARD_ALERT_THRESHOLD)

    for frame_result in det.iter_video(video_path, max_frames=max_frames):
        if not _process_frame(det, hs, disp, alerts, frame_result, window):
            break
    _finish_stream(window)


def _process_frame(
    det: BiomedicalDetector,
    hs: HazardScoring,
    disp: DisposalEngine,
    alerts: AlertManager,
    frame_result,
    window: bool,
) -> bool:
    """Returns False if user pressed Q in OpenCV window."""
    per = hs.score_frame(frame_result)
    max_h = hs.frame_max_hazard(per)
    tier, rec = disp.recommend(max_h)
    ev = alerts.evaluate(max_h, per)
    if ev:
        print(f"\rALERT [{ev.level}]: {ev.message}", file=sys.stderr)

    line = (
        f"det={frame_result.num_detections} hazard={max_h:.2f} {tier.value} "
        f"| {frame_result.inference_ms:.1f}ms"
    )
    print(f"\r{line}", end="", flush=True)

    if window:
        vis = det.annotate_frame(frame_result)
        cv2.putText(
            vis,
            f"Max hazard: {max_h:.1f} | {tier.value}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if max_h >= HAZARD_ALERT_THRESHOLD else (0, 255, 0),
            2,
        )
        cv2.imshow("Biomedical Waste — press Q to quit", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
    return True


def _finish_stream(window: bool) -> None:
    print()
    if window:
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Biomedical waste YOLOv8 detection + hazard scoring")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--source", type=Path, help="Image or video file")
    p.add_argument("--webcam", action="store_true", help="Use default webcam")
    p.add_argument("--camera", type=int, default=0, help="Webcam index if --webcam")
    p.add_argument("--no-window", action="store_true", help="Do not open OpenCV window (stream only)")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    if not args.weights.is_file():
        print(
            "Weights not found. Train first:\n"
            "  python -m biomedical_waste.scripts.train_cli --data data/data.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    show_win = not args.no_window

    if args.webcam:
        run_webcam(args.camera, args.weights, args.conf, show_win, args.max_frames)
    elif args.source:
        s = args.source
        suf = s.suffix.lower()
        if suf in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            run_image(s, args.weights, args.conf, show_win)
        else:
            run_video(s, args.weights, args.conf, show_win, args.max_frames)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
