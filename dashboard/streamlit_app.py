"""
Streamlit dashboard: biomedical waste detection with real-time hazard scoring and alerts.

Run from project root:
  streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from biomedical_waste.alerts import AlertManager, AlertEvent
from biomedical_waste.config import (
    DEFAULT_WEIGHTS,
    HAZARD_ALERT_THRESHOLD,
)
from biomedical_waste.detection import BiomedicalDetector
from biomedical_waste.disposal import DisposalEngine
from biomedical_waste.hazard import HazardScoring


@st.cache_resource
def _load_detector(weights_str: str, conf: float) -> BiomedicalDetector:
    """Cache detector to avoid reloading."""
    return BiomedicalDetector(weights=Path(weights_str), conf=conf)


def _init_state() -> None:
    """Initialize Streamlit session state."""
    state_vars = {
        "total_detections": 0,
        "class_counts": {},
        "hazard_series": [],
        "alert_count": 0,
        "last_alerts": [],
    }
    for k, v in state_vars.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_disposal_guidance() -> dict:
    """Get disposal guidance for all risk tiers."""
    return {
        "Critical": {
            "emoji": "🚨",
            "color": "#FF1744",
            "actions": [
                "🔴 Secure contaminated area immediately",
                "🔴 Use full PPE (gloves, mask, gown, eye protection)",
                "🔴 Notify safety officer and facility management",
                "🔴 Use biohazard/sharps containers with red labels",
                "🔴 Arrange emergency biological waste pickup",
            ]
        },
        "High": {
            "emoji": "⚠️",
            "color": "#FF6F00",
            "actions": [
                "🟠 Place in labeled sharps or biohazard containers",
                "🟠 Use appropriate PPE (gloves, mask minimum)",
                "🟠 Expedite pickup within 24-48 hours",
                "🟠 Ensure proper segregation per protocol",
                "🟠 Document disposal chain of custody",
            ]
        },
        "Moderate": {
            "emoji": "⚡",
            "color": "#FBC02D",
            "actions": [
                "🟡 Segregate waste per protocol",
                "🟡 Use appropriate waste containers",
                "🟡 Standard PPE and handling procedures",
                "🟡 Maintain documented records",
                "🟡 Follow facility guidelines",
            ]
        },
        "Low": {
            "emoji": "✅",
            "color": "#4CAF50",
            "actions": [
                "🟢 Follow general waste disposal guidelines",
                "🟢 No special biohazard precautions needed",
                "🟢 Standard disposal procedures apply",
                "🟢 No urgency required",
            ]
        }
    }


def _record_frame(
    class_names: list,
    max_hazard: float,
    alert_mgr: AlertManager,
    per_det: list,
) -> None:
    """Record frame detections to session state for tracking."""
    st.session_state.total_detections += len(class_names)
    for c in class_names:
        st.session_state.class_counts[c] = st.session_state.class_counts.get(c, 0) + 1
    
    st.session_state.hazard_series.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "max_hazard": float(max_hazard),
        }
    )
    if len(st.session_state.hazard_series) > 200:
        st.session_state.hazard_series = st.session_state.hazard_series[-200:]

    ev = alert_mgr.evaluate(max_hazard, per_det)
    if ev:
        st.session_state.alert_count += 1
        st.session_state.last_alerts.append(
            {"t": ev.timestamp, "msg": ev.message, "level": ev.level}
        )
        if len(st.session_state.last_alerts) > 50:
            st.session_state.last_alerts = st.session_state.last_alerts[-50:]




def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Biomedical Waste Monitoring",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    # Initialize core components
    hs = HazardScoring()
    disp = DisposalEngine()
    disposal_guidance = _get_disposal_guidance()

    st.title("🏥 Biomedical Waste Detection & Monitoring")
    st.caption("YOLOv8 • Hazard Scoring • Real-time Alerts • Disposal Guidance")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        weights = st.text_input(
            "Model weights path",
            value=str(DEFAULT_WEIGHTS),
        )
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        thr = st.slider(
            "Alert hazard threshold",
            0.0, 10.0,
            float(HAZARD_ALERT_THRESHOLD),
            0.1,
        )
        st.session_state.alert_threshold = thr

    weights_path = Path(weights)
    model_ready = weights_path.is_file()

    if not model_ready:
        st.error(
            f"❌ Model weights not found: {weights_path}\n\n"
            f"Train first with:\n```bash\n"
            f"python -m biomedical_waste.scripts.train_cli --data data/data.yaml\n```"
        )
        st.stop()

    # Tabs
    tab_dash, tab_img, tab_live = st.tabs(
        ["📊 Dashboard", "🖼️ Image", "📹 Live"]
    )

    # ============ DASHBOARD TAB ============
    with tab_dash:
        st.subheader("📈 Session Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Detections", st.session_state.total_detections)
        c2.metric("Alerts", st.session_state.alert_count)

        last_hazard = (
            st.session_state.hazard_series[-1]['max_hazard']
            if st.session_state.hazard_series
            else 0.0
        )
        c3.metric("Last Hazard", f"{last_hazard:.2f}" if last_hazard else "—")
        c4.metric("Threshold", f"{thr:.1f}")

        # Risk indicator
        st.subheader("🎯 Current Risk Level")
        col_r, col_d = st.columns(2)

        with col_r:
            if last_hazard > 0:
                tier, _ = disp.recommend(last_hazard)
                guidance = disposal_guidance.get(tier.value, {})
                color = guidance.get("color", "#999")
                st.progress(min(1.0, last_hazard / 10.0))
                html = f'''<div style="background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>{guidance.get("emoji", "")} {tier.value}</h3>
                    <p>Hazard: {last_hazard:.1f}/10</p>
                </div>'''
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No detections yet")

        with col_d:
            st.markdown("**Disposal Methods**")
            if last_hazard > 0:
                tier, _ = disp.recommend(last_hazard)
                guidance = disposal_guidance.get(tier.value, {})
                for action in guidance.get("actions", []):
                    st.markdown(f"• {action}")
            else:
                st.info("Run detection to see guidance")

        # Charts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            if st.session_state.class_counts:
                df = pd.DataFrame(
                    list(st.session_state.class_counts.items()),
                    columns=["Class", "Count"],
                )
                fig = px.bar(df, x="Class", y="Count", title="Waste Classes")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Class data pending")

        with col_c2:
            if st.session_state.hazard_series:
                df = pd.DataFrame(st.session_state.hazard_series)
                fig = px.line(df, x="time", y="max_hazard", title="Hazard Trend", markers=True)
                fig.add_hline(y=thr, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hazard data pending")

        if st.session_state.last_alerts:
            st.subheader("🚨 Recent Alerts")
            for a in reversed(st.session_state.last_alerts[-10:]):
                st.warning(f"[{a['level']}] {a['msg']}")

    # ============ IMAGE TAB ============
    with tab_img:
        st.markdown("Upload an image to detect waste and assess hazard.")
        upload = st.file_uploader("Select image", type=["jpg", "jpeg", "png", "bmp"])

        if upload and model_ready:
            try:
                data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

                if bgr is None:
                    st.error("Could not read image")
                else:
                    det = _load_detector(str(weights_path.resolve()), conf)
                    alert_mgr = AlertManager(hazard_threshold=thr)

                    with st.spinner("Running detection..."):
                        fr = det.predict_frame(bgr)
                        per = hs.score_frame(fr)
                        max_h = hs.frame_max_hazard(per)
                        tier, _ = disp.recommend(max_h)

                    _record_frame(fr.class_names, max_h, alert_mgr, per)

                    st.subheader("Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Detections", fr.num_detections)
                        st.metric("Max Hazard", f"{max_h:.2f}")
                    with c2:
                        guidance = disposal_guidance.get(tier.value, {})
                        color = guidance.get("color", "#999")
                        html = f'''<div style="background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3>{tier.value}</h3>
                        </div>'''
                        st.markdown(html, unsafe_allow_html=True)

                    # Annotated image
                    vis = det.annotate_frame(fr)
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

                    # Guidance
                    st.divider()
                    guidance = disposal_guidance.get(tier.value, {})
                    st.markdown(f"### {guidance.get('emoji', '')} {tier.value}")
                    for action in guidance.get("actions", []):
                        st.markdown(f"• {action}")

                    # Details table
                    if per:
                        st.subheader("Detection Details")
                        rows = [
                            {
                                "Class": p.class_name,
                                "Confidence": f"{p.confidence:.1%}",
                                "Hazard": f"{p.hazard_score:.2f}",
                            }
                            for p in per
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    if max_h >= thr:
                        st.error("🚨 ALERT: Hazard exceeds threshold!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ============ LIVE TAB ============
    with tab_live:
        st.markdown("Capture frames from browser webcam or upload video for analysis.")

        st.subheader("📷 Webcam")
        cam = st.camera_input("Capture")

        if cam is not None:
            try:
                raw = np.frombuffer(cam.getvalue(), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                if bgr is not None:
                    det = _load_detector(str(weights_path.resolve()), conf)
                    alert_mgr = AlertManager(hazard_threshold=thr)

                    with st.spinner("Analyzing..."):
                        fr = det.predict_frame(bgr)
                        per = hs.score_frame(fr)
                        max_h = hs.frame_max_hazard(per)
                        tier, _ = disp.recommend(max_h)

                    _record_frame(fr.class_names, max_h, alert_mgr, per)

                    vis = det.annotate_frame(fr)
                    color = (0, 0, 255) if max_h >= thr else (0, 255, 0)
                    cv2.putText(
                        vis,
                        f"Hazard {max_h:.1f} | {tier.value}",
                        (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2,
                    )

                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Detections", fr.num_detections)
                    c2.metric("Hazard", f"{max_h:.2f}")
                    c3.metric("Tier", tier.value)

                    st.divider()
                    guidance = disposal_guidance.get(tier.value, {})
                    st.markdown(f"### {guidance.get('emoji', '')} {tier.value}")
                    for action in guidance.get("actions", []):
                        st.markdown(f"• {action}")

                    if max_h >= thr:
                        st.error("🚨 ALERT!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Video upload
        st.divider()
        st.subheader("🎬 Video Upload")
        vup = st.file_uploader("Select video", type=["mp4", "avi", "mov", "mkv"])

        if vup is not None:
            try:
                suf = Path(vup.name).suffix or ".mp4"
                fd, tpath = tempfile.mkstemp(suffix=suf)
                os.close(fd)

                try:
                    Path(tpath).write_bytes(vup.getvalue())
                    det = _load_detector(str(weights_path.resolve()), conf)
                    alert_mgr = AlertManager(hazard_threshold=thr)

                    cap = cv2.VideoCapture(tpath)
                    if cap.isOpened():
                        st.info("Sampling first 24 frames...")
                        cols = st.columns(3)
                        fi = 0

                        while fi < 24:
                            ok, frame = cap.read()
                            if not ok:
                                break

                            with st.spinner(f"Frame {fi+1}/24..."):
                                fr = det.predict_frame(frame)
                                per = hs.score_frame(fr)
                                max_h = hs.frame_max_hazard(per)
                                tier, _ = disp.recommend(max_h)
                                _record_frame(fr.class_names, max_h, alert_mgr, per)

                                vis = det.annotate_frame(fr)
                                cv2.putText(
                                    vis,
                                    f"{max_h:.1f} {tier.value}",
                                    (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 200, 255),
                                    2,
                                )

                                cols[fi % 3].image(
                                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                    use_container_width=True,
                                    caption=f"Frame {fi+1}"
                                )

                            fi += 1

                        cap.release()
                        st.success("✅ Complete")
                    else:
                        st.error("Could not read video")

                finally:
                    try:
                        os.unlink(tpath)
                    except OSError:
                        pass

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
