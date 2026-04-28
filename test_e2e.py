#!/usr/bin/env python3
"""
Complete end-to-end verification script.
Tests all components working together in realistic scenarios.
"""

import sys
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add project to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_e2e_image_detection():
    """Test end-to-end image detection flow."""
    print("\n" + "="*60)
    print("TEST: End-to-End Image Detection")
    print("="*60)
    
    try:
        from biomedical_waste.detection import BiomedicalDetector
        from biomedical_waste.hazard import HazardScoring
        from biomedical_waste.disposal import DisposalEngine
        from biomedical_waste.alerts import AlertManager
        import cv2
        
        # Create mock image
        mock_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Save temporarily
        with NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, mock_image)
            temp_path = f.name
        
        try:
            # Initialize components
            detector = BiomedicalDetector()
            hazard = HazardScoring()
            disposal = DisposalEngine()
            alert_mgr = AlertManager()
            
            print(f"✓ Components initialized")
            
            # Run detection
            result = detector.predict_image(temp_path)
            print(f"✓ Detection completed: {result.num_detections} objects found")
            
            # Score hazards
            per_det = hazard.score_frame(result)
            max_h = hazard.frame_max_hazard(per_det)
            print(f"✓ Hazard scoring: max hazard = {max_h:.2f}/10")
            
            # Get disposal recommendation
            tier, rec = disposal.recommend(max_h)
            print(f"✓ Disposal recommendation: {tier.value}")
            
            # Check alerts
            alert = alert_mgr.evaluate(max_h, per_det)
            if alert:
                print(f"✓ Alert triggered: {alert.level}")
            else:
                print(f"✓ No alert (hazard below threshold)")
            
            print("\n✓ End-to-end image detection: PASS")
            return True
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"\n✗ End-to-end image detection: FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_e2e_multiple_detections():
    """Test handling multiple detections in a frame."""
    print("\n" + "="*60)
    print("TEST: Multiple Detections Handling")
    print("="*60)
    
    try:
        from biomedical_waste.detection import DetectionFrameResult
        from biomedical_waste.hazard import HazardScoring, PerDetectionHazard
        import numpy as np
        
        hs = HazardScoring()
        
        # Create mock result with 3 detections
        mock_result = DetectionFrameResult(
            frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
            boxes_xyxy=np.array([
                [10, 10, 100, 100],
                [150, 150, 250, 250],
                [300, 300, 400, 400],
            ], dtype=np.float32),
            class_ids=[0, 1, 3],  # General, Infectious, Sharps
            confidences=[0.9, 0.85, 0.95],
            class_names=["General", "Infectious", "Sharps"],
            inference_ms=25.0,
        )
        
        # Score frame
        per_det = hs.score_frame(mock_result)
        max_h = hs.frame_max_hazard(per_det)
        
        print(f"✓ Scored {len(per_det)} detections")
        for i, det in enumerate(per_det):
            print(f"  {i+1}. {det.class_name}: {det.hazard_score:.2f}/10")
        
        print(f"✓ Max hazard: {max_h:.2f}/10")
        max_det = max(per_det, key=lambda x: x.hazard_score)
        print(f"✓ Highest risk item: {max_det.class_name}")
        
        assert max_h == max(p.hazard_score for p in per_det), "Max hazard mismatch"
        
        print("\n✓ Multiple detections handling: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Multiple detections handling: FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_e2e_alert_escalation():
    """Test alert escalation with streaks."""
    print("\n" + "="*60)
    print("TEST: Alert Escalation & Streaks")
    print("="*60)
    
    try:
        from biomedical_waste.alerts import AlertManager
        
        alert_mgr = AlertManager(
            hazard_threshold=6.5,
            streak_threshold=3,
            cooldown_sec=0.1
        )
        
        # Simulate multiple high-hazard frames
        hazards = [7.0, 7.5, 8.0]  # Above threshold
        
        for i, h in enumerate(hazards):
            ev = alert_mgr.evaluate(h, [])
            level = ev.level if ev else None
            expected = "warning" if i < 2 else "critical"  # Escalate on frame 3
            print(f"  Frame {i+1}: hazard={h:.1f}, level={level}")
        
        # Last alert should be critical
        final_alert = alert_mgr.evaluate(7.0, [])
        assert final_alert.level == "critical", "Should escalate to critical"
        
        print(f"✓ Alert escalated to critical after {alert_mgr._streak} frames")
        
        # Reset and test recovery
        alert_mgr.reset_streak()
        ev = alert_mgr.evaluate(3.0, [])  # Below threshold
        assert ev is None, "Should not alert below threshold"
        assert alert_mgr._streak == 0, "Streak should reset"
        
        print(f"✓ Streak properly reset")
        print("\n✓ Alert escalation: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Alert escalation: FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_e2e_disposal_tier_mapping():
    """Test hazard score to disposal tier mapping."""
    print("\n" + "="*60)
    print("TEST: Hazard to Disposal Tier Mapping")
    print("="*60)
    
    try:
        from biomedical_waste.disposal import DisposalEngine
        
        disp = DisposalEngine()
        
        test_cases = [
            (1.0, "Low"),
            (2.5, "Low"),
            (3.5, "Moderate"),
            (5.0, "Moderate"),
            (6.0, "High"),
            (7.5, "High"),
            (8.0, "Critical"),
            (9.5, "Critical"),
        ]
        
        all_pass = True
        for score, expected_tier in test_cases:
            tier, _ = disp.recommend(score)
            status = "✓" if tier.value == expected_tier else "✗"
            print(f"  {status} Score {score}: {tier.value} (expected {expected_tier})")
            if tier.value != expected_tier:
                all_pass = False
        
        assert all_pass, "Some tier mappings failed"
        
        print("\n✓ Disposal tier mapping: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Disposal tier mapping: FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_e2e_config_consistency():
    """Test configuration consistency."""
    print("\n" + "="*60)
    print("TEST: Configuration Consistency")
    print("="*60)
    
    try:
        from biomedical_waste.config import (
            CLASS_NAMES,
            BASE_RISK,
            HAZARD_ALERT_THRESHOLD,
            DEFAULT_CONF,
            DEFAULT_IMGSZ,
        )
        
        # Check class names and risk mapping
        assert len(CLASS_NAMES) == 4, "Should have 4 waste classes"
        print(f"✓ Classes defined: {', '.join(CLASS_NAMES)}")
        
        # Check risk scores
        assert all(name in BASE_RISK for name in CLASS_NAMES), "All classes have risk scores"
        assert all(0 <= score <= 10 for score in BASE_RISK.values()), "Risk scores in 0-10 range"
        print(f"✓ Risk scores defined for all classes")
        
        # Check thresholds
        assert 0 <= HAZARD_ALERT_THRESHOLD <= 10, "Alert threshold in valid range"
        assert 0 < DEFAULT_CONF < 1, "Confidence threshold in valid range"
        assert DEFAULT_IMGSZ > 0, "Image size positive"
        print(f"✓ All thresholds valid")
        
        print(f"\nConfiguration Summary:")
        print(f"  Classes: {list(CLASS_NAMES)}")
        print(f"  Risk scores: {BASE_RISK}")
        print(f"  Alert threshold: {HAZARD_ALERT_THRESHOLD}")
        print(f"  Confidence threshold: {DEFAULT_CONF}")
        print(f"  Image size: {DEFAULT_IMGSZ}x{DEFAULT_IMGSZ}")
        
        print("\n✓ Configuration consistency: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration consistency: FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all end-to-end tests."""
    print("="*60)
    print("Biomedical Waste Detection System - E2E Verification")
    print("="*60)
    
    tests = [
        ("Configuration", test_e2e_config_consistency),
        ("Image Detection", test_e2e_image_detection),
        ("Multiple Detections", test_e2e_multiple_detections),
        ("Alert Escalation", test_e2e_alert_escalation),
        ("Disposal Mapping", test_e2e_disposal_tier_mapping),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name}: CRASH - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("END-TO-END TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("✓ ALL END-TO-END TESTS PASSED!")
        print("="*60)
        print("\nSystem is fully functional and ready for production use.")
        print("\nNext steps:")
        print("  1. Start dashboard: python startup.py")
        print("  2. Upload images or use webcam")
        print("  3. Monitor detections and alerts")
        print("  4. View disposal recommendations")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
