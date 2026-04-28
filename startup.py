#!/usr/bin/env python3
"""
Startup script for Biomedical Waste Detection System.
Handles environment setup and launches the appropriate interface.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional


def check_environment() -> bool:
    """Check if all required packages are installed."""
    print("Checking environment...")
    required = [
        "streamlit",
        "ultralytics",
        "cv2",
        "numpy",
        "pandas",
        "plotly",
    ]
    
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"✗ Missing modules: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All required modules installed")
    return True


def check_model_weights() -> bool:
    """Check if model weights exist."""
    weights_path = Path("runs/detect/biomedical_waste_30ep/weights/best.pt")
    if not weights_path.exists():
        print(f"✗ Model weights not found: {weights_path}")
        print("Train first with:")
        print("  python -m biomedical_waste.scripts.train_cli --data data/data.yaml")
        return False
    
    print(f"✓ Model weights found: {weights_path}")
    return True


def run_dashboard() -> int:
    """Launch Streamlit dashboard."""
    print("\nStarting Biomedical Waste Detection Dashboard...")
    print("Opening http://localhost:8501 in browser...")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard/streamlit_app.py"
        ], check=False)
        return 0
    except KeyboardInterrupt:
        print("\n✓ Dashboard stopped")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def run_cli_detection(source: str, weights: Optional[str] = None) -> int:
    """Run command-line detection."""
    args = [sys.executable, "run_detection.py", "--source", source]
    if weights:
        args.extend(["--weights", weights])
    
    try:
        subprocess.run(args, check=False)
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def run_webcam() -> int:
    """Run real-time webcam detection."""
    print("Starting webcam detection...")
    print("Press 'q' to quit\n")
    
    try:
        subprocess.run([sys.executable, "run_detection.py", "--webcam"], check=False)
        return 0
    except KeyboardInterrupt:
        print("\n✓ Webcam stopped")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def run_tests() -> int:
    """Run system tests."""
    print("Running system tests...\n")
    
    try:
        subprocess.run([sys.executable, "test_system.py"], check=True)
        return 0
    except subprocess.CalledProcessError:
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def main():
    """Main startup logic."""
    parser = argparse.ArgumentParser(
        description="Biomedical Waste Detection System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python startup.py                      # Launch dashboard (default)
  python startup.py --webcam             # Real-time webcam detection
  python startup.py --image image.jpg    # Detect in image
  python startup.py --video video.mp4    # Detect in video
  python startup.py --test               # Run system tests
        """
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=True,
        help="Launch Streamlit dashboard (default)"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time webcam detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        metavar="PATH",
        help="Detect on image file"
    )
    parser.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Detect on video file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system tests"
    )
    parser.add_argument(
        "--weights",
        type=str,
        metavar="PATH",
        help="Path to model weights"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Biomedical Waste Detection System")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        return 1
    
    # Check model weights
    if not check_model_weights():
        if not args.test:
            return 1
    
    # Run tests
    if args.test:
        return run_tests()
    
    # Run webcam
    if args.webcam:
        return run_webcam()
    
    # Run image detection
    if args.image:
        if not Path(args.image).exists():
            print(f"✗ Image not found: {args.image}")
            return 1
        return run_cli_detection(args.image, args.weights)
    
    # Run video detection
    if args.video:
        if not Path(args.video).exists():
            print(f"✗ Video not found: {args.video}")
            return 1
        return run_cli_detection(args.video, args.weights)
    
    # Default: run dashboard
    return run_dashboard()


if __name__ == "__main__":
    sys.exit(main())
