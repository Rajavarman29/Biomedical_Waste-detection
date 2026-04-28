#!/usr/bin/env python3
"""Test script to verify the entire system works end-to-end."""

from pathlib import Path
import sys

# Add project to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    try:
        from biomedical_waste.alerts import AlertManager, AlertEvent
        from biomedical_waste.config import (
            DEFAULT_WEIGHTS,
            HAZARD_ALERT_THRESHOLD,
            CLASS_NAMES,
            BASE_RISK,
        )
        from biomedical_waste.detection import BiomedicalDetector, DetectionFrameResult
        from biomedical_waste.disposal import DisposalEngine, DisposalTier
        from biomedical_waste.hazard import HazardScoring, PerDetectionHazard
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_detector():
    """Test detector initialization."""
    print("\nTesting detector...")
    try:
        from biomedical_waste.detection import BiomedicalDetector
        from biomedical_waste.config import DEFAULT_WEIGHTS
        
        detector = BiomedicalDetector(weights=Path(DEFAULT_WEIGHTS))
        print(f"✓ Detector initialized")
        print(f"  Model: {str(DEFAULT_WEIGHTS)}")
        return True
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return False


def test_hazard_scoring():
    """Test hazard scoring."""
    print("\nTesting hazard scoring...")
    try:
        from biomedical_waste.hazard import HazardScoring
        from biomedical_waste.detection import DetectionFrameResult
        import numpy as np
        
        hs = HazardScoring()
        
        # Create a mock detection result
        mock_result = DetectionFrameResult(
            frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
            boxes_xyxy=np.array([[10, 10, 100, 100]], dtype=np.float32),
            class_ids=[1],  # Infectious
            confidences=[0.9],
            class_names=["Infectious"],
            inference_ms=25.0,
        )
        
        per = hs.score_frame(mock_result)
        max_h = hs.frame_max_hazard(per)
        
        print(f"✓ Hazard scoring works")
        print(f"  Mock detection: {per[0].class_name}, confidence={per[0].confidence:.2f}")
        print(f"  Hazard score: {per[0].hazard_score:.2f}/10")
        print(f"  Max hazard: {max_h:.2f}/10")
        return True
    except Exception as e:
        print(f"✗ Hazard scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disposal_engine():
    """Test disposal engine."""
    print("\nTesting disposal engine...")
    try:
        from biomedical_waste.disposal import DisposalEngine
        
        disp = DisposalEngine()
        
        test_scores = [2.0, 5.0, 7.5, 9.0]
        print(f"✓ Disposal engine initialized")
        for score in test_scores:
            tier, rec = disp.recommend(score)
            print(f"  Score {score}: {tier.value} - {rec[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Disposal engine failed: {e}")
        return False


def test_alerts():
    """Test alert manager."""
    print("\nTesting alert manager...")
    try:
        from biomedical_waste.alerts import AlertManager
        
        alert_mgr = AlertManager(hazard_threshold=6.5)
        
        # Test no alert (below threshold)
        ev1 = alert_mgr.evaluate(3.0, [])
        assert ev1 is None, "Should not trigger alert below threshold"
        
        # Test alert (above threshold)
        ev2 = alert_mgr.evaluate(7.0, [])
        assert ev2 is not None, "Should trigger alert above threshold"
        assert ev2.level == "warning", "Should be warning level"
        
        print(f"✓ Alert manager works")
        print(f"  Alert triggered: {ev2.message}")
        return True
    except Exception as e:
        print(f"✗ Alert manager failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Biomedical Waste Detection System - Full Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Detector", test_detector()))
    results.append(("Hazard Scoring", test_hazard_scoring()))
    results.append(("Disposal Engine", test_disposal_engine()))
    results.append(("Alert Manager", test_alerts()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nTo start the dashboard:")
        print("  streamlit run dashboard/streamlit_app.py")
        print("\nTo run command-line detection:")
        print("  python run_detection.py --source image.jpg")
        return 0
    else:
        print("✗ Some tests failed. Fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
