#  Biomedical Waste Detection System

> A real-time YOLOv8-based computer vision system for automatic detection, classification, and risk assessment of biomedical waste in healthcare environments — with an interactive Streamlit dashboard for live monitoring.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20Nano-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

---

##  Problem Statement

Improper segregation of biomedical waste is a serious public health hazard in hospitals and clinics. Manual identification is error-prone, slow, and dependent on trained staff. This system provides automated, real-time detection of hazardous medical waste using a lightweight computer vision model — enabling instant alerts and data-driven monitoring without requiring specialized personnel.

---

##  Key Features

-  **Real-Time Detection** — YOLOv8 Nano inference at 30+ FPS on GPU, ~8 FPS on CPU
-  **4-Class Hazard Classification** — General, Infectious, Pathological, Sharps
-  **Intelligent Hazard Scoring** — Risk score (0–10) combining class base risk + detection confidence
-  **Alert Management** — Streak-based escalation and cooldown system for real-time notifications
-  **Interactive Dashboard** — Streamlit UI with live webcam feed, charts, and alert history
-  **Multi-Source Input** — Supports images, video files, and live webcam streams
-  **Configurable Thresholds** — All risk levels, confidence thresholds, and alert rules centralized in `config.py`
-  **Extensive Logging** — Detection history and alert logs for post-analysis
-  **Export Ready** — Model exportable to ONNX, TFLite for edge deployment

---

##  Waste Classes & Risk Levels

| Class | Risk Level | Hazard Score | Examples |
|---|---|---|---|
|  **General** | Low | 1–3 | Non-hazardous packaging, paper |
|  **Infectious** | High | 6–8 | Blood-contaminated materials, cultures |
|  **Pathological** | High | 7–9 | Tissues, organs, anatomical waste |
|  **Sharps** | Critical | 9–10 | Needles, blades, lancets, scalpels |

---

##  Hazard Scoring Algorithm

```
hazard_score = (base_risk × confidence) / 10

Alert triggered  if: hazard_score ≥ 6.5
Streak escalation if: high_risk_frames ≥ 5 consecutive frames
Cooldown between alerts: 2.0 seconds
```

---

##  Architecture

```
Image / Video / Webcam Input
         ↓
  YOLOv8 Nano Inference (640×640)
         ↓
  Bounding Box + Class + Confidence
         ↓
  ┌──────────────────────────────┐
  │     Hazard Scoring Engine    │
  │  base_risk × confidence / 10 │
  └──────────────────────────────┘
         ↓
  ┌──────────────┐   ┌─────────────────────┐
  │ Alert Manager│   │  Disposal Recommender│
  │ (escalation) │   │  (per-class guidance)│
  └──────────────┘   └─────────────────────┘
         ↓
  Streamlit Dashboard + Logs
```

### Project Structure

```
biomedical-waste-detection/
│
├── biomedical_waste/
│   ├── config.py              # Centralized config (thresholds, class names, paths)
│   ├── detection/
│   │   └── detector.py        # YOLOv8 inference for images and video streams
│   ├── training/
│   │   └── train.py           # Full training pipeline
│   ├── hazard/
│   │   └── scoring.py         # Hazard score computation
│   ├── alerts/
│   │   └── alert_manager.py   # Streak-based alerting with cooldown
│   ├── disposal/
│   │   └── recommendations.py # Per-class disposal guidance
│   └── scripts/
│       ├── train_cli.py        # Training CLI entry point
│       └── prepare_dataset.py  # Dataset utilities
│
├── dashboard/
│   └── streamlit_app.py       # Interactive monitoring dashboard
│
├── data/
│   └── medical-waste-2/       # Roboflow dataset (train/valid/test)
│       └── data.yaml
│
├── runs/detect/               # Training outputs and model weights
├── run_detection.py           # Detection CLI entry point
├── monitor_training.py        # Real-time training monitor
└── requirements.txt
```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Object Detection | YOLOv8 Nano (Ultralytics) |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Plotly, Streamlit |
| Dataset | Roboflow "medical-waste-2" (CC BY 4.0) |

---

##  How to Run

### Prerequisites
- Python 3.8+
- GPU (CUDA) recommended; CPU supported

### Installation

```bash
git clone https://github.com/rajavarman/biomedical-waste-detection.git
cd biomedical-waste-detection

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### Run Detection

```bash
python run_detection.py --source path/to/image.jpg      # Image
python run_detection.py --source path/to/video.mp4      # Video
python run_detection.py --source 0                      # Live webcam
```

### Launch Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501` for live webcam detection, image upload, detection trends, and alert history.

### Train a Custom Model

```bash
python -m biomedical_waste.scripts.train_cli \
  --data data/data.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640
```

---

##  Dashboard Features

| Panel | Description |
|---|---|
| Live Webcam | Real-time bounding box overlay with hazard score |
| Image Upload | Analyze single images with per-detection breakdown |
| Video Analysis | Frame-by-frame detection on uploaded video |
| Metrics | Detection count, confidence, hazard score distribution |
| Alert History | Log of triggered alerts with timestamps |
| Config Panel | Adjust confidence thresholds and alert settings live |

---

##  Screenshots

| Live Detection | Dashboard | Alert History |
|---|---|---|
| ![Detection](screenshots/detection.png) | ![Dashboard](screenshots/dashboard.png) | ![Alerts](screenshots/alerts.png) |

> _Add screenshots to `/screenshots` after running._

---

##  Performance

| Hardware | FPS | Latency |
|---|---|---|
| NVIDIA RTX 3080 | 40+ | ~25ms |
| NVIDIA RTX 2060 | 20–30 | ~40ms |
| Intel i7 CPU | 8–10 | ~100ms |

**Model**: YOLOv8 Nano | **Input**: 640×640px | **Trained**: 30 epochs on Roboflow dataset

---

##  Configuration Reference

All settings in `biomedical_waste/config.py`:

| Setting | Default | Description |
|---|---|---|
| `HAZARD_ALERT_THRESHOLD` | `6.5` | Minimum hazard score to trigger alert |
| `HIGH_RISK_STREAK_ALERT` | `5` | Frames of consecutive high-risk for escalation |
| `ALERT_COOLDOWN_SEC` | `2.0` | Seconds between duplicate alerts |
| `DEFAULT_CONF` | `0.25` | Detection confidence threshold |
| `DEFAULT_IMGSZ` | `640` | Model input resolution |

---

##  Privacy

All processing is fully local. No images or video are uploaded to any external service. Detection logs are stored only in the local `/runs` directory.

---

##  Author

**Rajavarman M** — B.Tech AI & Data Science, Rajalakshmi Institute of Technology  
📧 rajavarman419@gmail.com | 🔗 [LinkedIn](https://linkedin.com/in/rajavarman)