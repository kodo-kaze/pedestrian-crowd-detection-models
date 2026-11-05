#!/usr/bin/env python3
# =============================================================================
# CROWD DETECTION V1 — REAL-TIME FASTAPI CCTV STREAMING SERVER
#
# Usage:
#     python stream_server.py [options]
#
# Examples:
#     python stream_server.py --source 0                         # Live Webcam
#     python stream_server.py --source input/sample_video.mp4   # Loop Video File
#     python stream_server.py --source rtsp://ip:554/live       # Live CCTV / IP Camera
#     python stream_server.py --port 8000 --stride 2            # Custom port & 2x speed
#
# Architecture:
#     [CCTV Camera / Webcam / RTSP / Video]
#                  │
#                  ▼ (Thread 1: Zero-latency bufferless reader)
#          Latest Raw Frame
#                  │
#                  ▼ (Thread 2: Real-time AI Worker)
#          YOLO26m Person Detector
#                  │
#          21 Spatial & Density Features
#                  │
#          Random Forest Classifier (Anti-flicker temporal smoothing)
#                  │
#          Visual Annotation (Bounding boxes + Glassmorphism HUD)
#                  │
#                  ▼
#          Latest Encoded JPEG Frame + Telemetry State
#                  │
#                  ▼ (Thread 3: FastAPI Async Broadcaster)
#     ┌────────────────────────┴────────────────────────┐
#     ▼                                                 ▼
#  GET /video_feed (MJPEG Stream)             GET / (Live Web Dashboard)
#  GET /stats      (JSON Telemetry)           GET /health (Healthcheck)
# =============================================================================

import argparse
from collections import deque
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import numpy as np
import pandas as pd
from ultralytics import YOLO
import uvicorn

# =============================================================================
# CONFIGURATION & MODEL RESOLUTION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "person_count",
    "avg_confidence",
    "min_confidence",
    "max_confidence",
    "total_bbox_area_ratio",
    "avg_bbox_area_ratio",
    "max_bbox_area_ratio",
    "avg_bbox_width_ratio",
    "avg_bbox_height_ratio",
    "people_per_megapixel",
    "average_center_distance",
    "average_nearest_neighbor_distance",
    "grid_00", "grid_01", "grid_02",
    "grid_10", "grid_11", "grid_12",
    "grid_20", "grid_21", "grid_22",
]


def resolve_model_path(filename: str) -> Path:
    """Finds model weights across standard project directories."""
    candidates = [
        SCRIPT_DIR / "models" / filename,
        SCRIPT_DIR / "pcp-model-api" / filename,
        SCRIPT_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return SCRIPT_DIR / "models" / filename


# =============================================================================
# FEATURE EXTRACTION ENGINE
# =============================================================================

def extract_features(result) -> Dict[str, Any]:
    """Computes the 21 scene-level spatial & density features."""
    image = result.orig_img
    height, width = image.shape[:2]
    image_area = width * height

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "person_count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "total_bbox_area_ratio": 0.0,
            "avg_bbox_area_ratio": 0.0,
            "max_bbox_area_ratio": 0.0,
            "avg_bbox_width_ratio": 0.0,
            "avg_bbox_height_ratio": 0.0,
            "people_per_megapixel": 0.0,
            "average_center_distance": 0.0,
            "average_nearest_neighbor_distance": 0.0,
            "grid_00": 0, "grid_01": 0, "grid_02": 0,
            "grid_10": 0, "grid_11": 0, "grid_12": 0,
            "grid_20": 0, "grid_21": 0, "grid_22": 0,
        }

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    person_count = len(boxes)
    avg_confidence = float(np.mean(confidences))
    min_confidence = float(np.min(confidences))
    max_confidence = float(np.max(confidences))

    bbox_widths = boxes[:, 2] - boxes[:, 0]
    bbox_heights = boxes[:, 3] - boxes[:, 1]
    bbox_areas = bbox_widths * bbox_heights

    bbox_area_ratios = bbox_areas / image_area
    total_bbox_area_ratio = float(np.sum(bbox_area_ratios))
    avg_bbox_area_ratio = float(np.mean(bbox_area_ratios))
    max_bbox_area_ratio = float(np.max(bbox_area_ratios))

    avg_bbox_width_ratio = float(np.mean(bbox_widths / width))
    avg_bbox_height_ratio = float(np.mean(bbox_heights / height))

    image_megapixels = image_area / 1_000_000
    people_per_megapixel = float(person_count / max(image_megapixels, 1e-6))

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.column_stack([centers_x / width, centers_y / height])

    if person_count >= 2:
        pairwise_distances = []
        for i in range(person_count):
            for j in range(i + 1, person_count):
                pairwise_distances.append(np.linalg.norm(centers[i] - centers[j]))
        average_center_distance = float(np.mean(pairwise_distances))
    else:
        average_center_distance = 0.0

    if person_count >= 2:
        nearest_distances = []
        for i in range(person_count):
            distances = np.linalg.norm(centers - centers[i], axis=1)
            distances[i] = np.inf
            nearest_distances.append(np.min(distances))
        average_nearest_neighbor_distance = float(np.mean(nearest_distances))
    else:
        average_nearest_neighbor_distance = 0.0

    grid_counts = np.zeros((3, 3), dtype=int)
    for cx, cy in centers:
        col = min(int(cx * 3), 2)
        row = min(int(cy * 3), 2)
        grid_counts[row, col] += 1

    return {
        "person_count": person_count,
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "total_bbox_area_ratio": total_bbox_area_ratio,
        "avg_bbox_area_ratio": avg_bbox_area_ratio,
        "max_bbox_area_ratio": max_bbox_area_ratio,
        "avg_bbox_width_ratio": avg_bbox_width_ratio,
        "avg_bbox_height_ratio": avg_bbox_height_ratio,
        "people_per_megapixel": people_per_megapixel,
        "average_center_distance": average_center_distance,
        "average_nearest_neighbor_distance": average_nearest_neighbor_distance,
        "grid_00": int(grid_counts[0, 0]),
        "grid_01": int(grid_counts[0, 1]),
        "grid_02": int(grid_counts[0, 2]),
        "grid_10": int(grid_counts[1, 0]),
        "grid_11": int(grid_counts[1, 1]),
        "grid_12": int(grid_counts[1, 2]),
        "grid_20": int(grid_counts[2, 0]),
        "grid_21": int(grid_counts[2, 1]),
        "grid_22": int(grid_counts[2, 2]),
    }


# =============================================================================
# VISUAL HUD ANNOTATION ENGINE
# =============================================================================

def draw_hud_and_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    is_crowd: bool,
    crowd_prob: float,
    person_count: int,
    density_per_mp: float,
    nn_distance: float,
    fps: float
) -> np.ndarray:
    """Renders individual person bounding boxes and a top analytical HUD banner."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Color palette (BGR)
    if is_crowd:
        theme_color = (34, 48, 220)       # Crimson / Red
        badge_bg = (24, 30, 180)
        status_text = "CROWD ALERT"
    else:
        theme_color = (60, 180, 75)       # Emerald / Green
        badge_bg = (40, 140, 55)
        status_text = "NORMAL DENSITY"

    # 1. Bounding boxes
    if boxes is not None and len(boxes) > 0:
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_color, 2)
            tag = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(th + 4, y1)), theme_color, -1)
            cv2.putText(annotated, tag, (x1 + 2, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Glassmorphism Top HUD Banner
    hud_height = max(70, int(h * 0.10))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.line(annotated, (0, hud_height), (w, hud_height), theme_color, 2)

    scale_title = max(0.65, min(1.0, w / 1280))
    scale_sub = max(0.45, min(0.70, w / 1280))

    badge_width = int(220 * scale_title)
    badge_height = int(hud_height * 0.55)
    cv2.rectangle(annotated, (14, 10), (14 + badge_width, 10 + badge_height), badge_bg, -1)
    cv2.rectangle(annotated, (14, 10), (14 + badge_width, 10 + badge_height), theme_color, 2)

    cv2.putText(
        annotated,
        f"[{status_text}]",
        (22, int(10 + badge_height * 0.70)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_title,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Probability Gauge
    prob_bar_x = 22 + badge_width
    prob_bar_w = int(140 * scale_title)
    prob_bar_h = max(8, int(badge_height * 0.35))
    prob_bar_y = 10 + int(badge_height * 0.30)

    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + prob_bar_w, prob_bar_y + prob_bar_h), (50, 50, 60), -1)
    fill_w = int(prob_bar_w * np.clip(crowd_prob, 0.0, 1.0))
    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + fill_w, prob_bar_y + prob_bar_h), theme_color, -1)
    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + prob_bar_w, prob_bar_y + prob_bar_h), (120, 120, 130), 1)

    cv2.putText(
        annotated,
        f"{crowd_prob * 100:.1f}%",
        (prob_bar_x + prob_bar_w + 10, prob_bar_y + prob_bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_sub,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )

    # Telemetry Sub-bar
    metrics_str = (
        f"People: {person_count:3d}   |   "
        f"Density: {density_per_mp:5.1f}/MP   |   "
        f"NN-Dist: {nn_distance:0.3f}   |   "
        f"Live FPS: {fps:4.1f}"
    )
    cv2.putText(
        annotated,
        metrics_str,
        (20, hud_height - max(12, int(hud_height * 0.18))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_sub,
        (210, 215, 220),
        1,
        cv2.LINE_AA
    )

    return annotated


# =============================================================================
# ZERO-LATENCY RTSP / CAMERA CAPTURE THREAD
# =============================================================================

class RTSPCameraStream:
    """
    Decoupled background capture thread.
    Continuously retrieves the absolute freshest frame to prevent RTSP queue buildup.
    """
    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.is_video_file = isinstance(self.source, str) and not (self.source.startswith("rtsp://") or self.source.startswith("http://"))
        self.cap = cv2.VideoCapture(self.source)
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.connected = False

        if self.cap.isOpened():
            self.connected = True
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            if not self.cap.isOpened():
                self.connected = False
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.source)
                continue

            ret, frame = self.cap.read()
            if ret:
                self.connected = True
                with self.lock:
                    self.latest_frame = frame
            else:
                if self.is_video_file:
                    # Seamlessly loop test video files
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.connected = False
                    time.sleep(0.5)

            # Prevent 100% CPU usage in read thread
            time.sleep(0.005)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# =============================================================================
# REAL-TIME AI INFERENCE & STREAM BROADCASTER
# =============================================================================

class CrowdStreamBroadcaster:
    """
    Runs the YOLO + Feature + Random Forest pipeline in a continuous loop.
    Stores the latest annotated JPEG frame and telemetry stats for multi-client distribution.
    """
    def __init__(self, source, conf: float = 0.25, iou: float = 0.70, stride: int = 1, smooth: int = 10):
        self.source = source
        self.conf = conf
        self.iou = iou
        self.stride = max(1, stride)
        self.smooth = max(1, smooth)

        # Load models
        yolo_path = resolve_model_path("best.pt")
        rf_path = resolve_model_path("crowd_random_forest_v1.pkl")

        print(f"Loading YOLO model from: {yolo_path}")
        self.yolo_model = YOLO(str(yolo_path))

        print(f"Loading Random Forest model from: {rf_path}")
        with open(rf_path, "rb") as f:
            self.rf_model = pickle.load(f)
        self.rf_model.n_jobs = 1

        self.camera = RTSPCameraStream(self.source)

        # Thread-safe broadcast state
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.stats = {
            "status": "INITIALIZING",
            "is_crowd": False,
            "crowd_probability": 0.0,
            "people_count": 0,
            "density_per_megapixel": 0.0,
            "nearest_neighbor_distance": 0.0,
            "fps": 0.0,
            "camera_connected": False,
            "uptime_seconds": 0,
        }

        self.running = True
        self.start_time = time.time()
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

    def _inference_loop(self):
        prob_history = deque(maxlen=self.smooth)
        frame_counter = 0
        last_time = time.time()

        # Cache last detections for skipped frames
        cached_boxes = np.empty((0, 4))
        cached_confs = np.empty(0)
        cached_features = {
            "person_count": 0,
            "people_per_megapixel": 0.0,
            "average_nearest_neighbor_distance": 0.0
        }
        cached_prob = 0.0
        cached_is_crowd = False

        while self.running:
            frame = self.camera.read()
            if frame is None:
                with self.lock:
                    self.stats["camera_connected"] = False
                time.sleep(0.05)
                continue

            frame_counter += 1
            now = time.time()
            fps_instant = 1.0 / max(now - last_time, 1e-4)
            last_time = now

            # Stride logic: only run heavy models on every Nth frame
            if frame_counter % self.stride == 0:
                # 1. YOLO
                results = self.yolo_model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    classes=[0],
                    max_det=500,
                    verbose=False
                )
                result = results[0]

                # 2. Features
                features = extract_features(result)
                cached_features = features

                # 3. Random Forest
                feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
                raw_prob = float(self.rf_model.predict_proba(feature_df)[0][1])

                # 4. Anti-flicker temporal smoothing
                prob_history.append(raw_prob)
                smoothed_prob = float(np.mean(prob_history))
                cached_prob = smoothed_prob
                cached_is_crowd = smoothed_prob >= 0.50

                cached_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
                cached_confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)

            # 5. Draw Annotations & HUD
            annotated_frame = draw_hud_and_boxes(
                frame=frame,
                boxes=cached_boxes,
                confidences=cached_confs,
                is_crowd=cached_is_crowd,
                crowd_prob=cached_prob,
                person_count=cached_features["person_count"],
                density_per_mp=cached_features["people_per_megapixel"],
                nn_distance=cached_features["average_nearest_neighbor_distance"],
                fps=fps_instant
            )

            # 6. Encode to JPEG for network broadcasting
            success, jpeg_buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                jpeg_bytes = jpeg_buffer.tobytes()
                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    self.stats = {
                        "status": "CROWD ALERT" if cached_is_crowd else "NORMAL",
                        "is_crowd": cached_is_crowd,
                        "crowd_probability": round(cached_prob, 4),
                        "people_count": cached_features["person_count"],
                        "density_per_megapixel": round(cached_features["people_per_megapixel"], 2),
                        "nearest_neighbor_distance": round(cached_features["average_nearest_neighbor_distance"], 4),
                        "fps": round(fps_instant, 1),
                        "camera_connected": self.camera.connected,
                        "uptime_seconds": int(time.time() - self.start_time),
                    }

    def get_latest_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.stats)

    def stop(self):
        self.running = False
        self.camera.stop()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Crowd Detection & Intelligence Live Streaming API",
    description="Real-time CCTV/RTSP Crowd Ingestion, Inference & WebRTC/MJPEG Re-broadcast",
    version="1.0.0"
)

broadcaster: Optional[CrowdStreamBroadcaster] = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Returns the executive dark-mode real-time monitoring dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crowd Intelligence Stream</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #0f1117; color: #e2e8f0; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #232734; }
        .header h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; color: #ffffff; }
        .header p { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
        .status-pill { padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }
        .status-normal { background: rgba(56, 161, 105, 0.2); color: #48bb78; border: 1px solid #38a169; }
        .status-crowd { background: rgba(229, 62, 62, 0.25); color: #f56565; border: 1px solid #e53e3e; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-box { background: #161922; border-radius: 12px; overflow: hidden; border: 1px solid #232734; display: flex; align-items: center; justify-content: center; min-height: 480px; position: relative; }
        .video-box img { width: 100%; height: auto; display: block; }
        .metrics-panel { display: flex; flex-direction: column; gap: 14px; }
        .metric-card { background: #161922; padding: 18px; border-radius: 12px; border: 1px solid #232734; }
        .metric-card .title { font-size: 0.78rem; text-transform: uppercase; color: #8892b0; font-weight: 600; letter-spacing: 0.05em; }
        .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-top: 6px; }
        .metric-card .sub { font-size: 0.75rem; color: #718096; margin-top: 4px; }
        .footer { margin-top: 24px; padding-top: 16px; border-top: 1px solid #232734; font-size: 0.8rem; color: #718096; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Crowd Detection & Density Monitoring</h1>
                <p>Live Ingestion & Processing Server &bull; YOLO26m + 21-Feature Random Forest</p>
            </div>
            <div id="statusBadge" class="status-pill status-normal">NORMAL</div>
        </div>

        <div class="grid">
            <div class="video-box">
                <img src="/video_feed" alt="Live Stream" />
            </div>

            <div class="metrics-panel">
                <div class="metric-card">
                    <div class="title">Detected Individuals</div>
                    <div class="value" id="kpiCount">--</div>
                    <div class="sub">Instantaneous bounding count</div>
                </div>

                <div class="metric-card">
                    <div class="title">Spatial Density</div>
                    <div class="value" id="kpiDensity">-- <span style="font-size: 1rem;">/ MP</span></div>
                    <div class="sub">People per megapixel area</div>
                </div>

                <div class="metric-card">
                    <div class="title">Crowd Probability</div>
                    <div class="value" id="kpiProb">--%</div>
                    <div class="sub">10-frame smoothed classification confidence</div>
                </div>

                <div class="metric-card">
                    <div class="title">Inference Throughput</div>
                    <div class="value" id="kpiFps">-- <span style="font-size: 1rem;">FPS</span></div>
                    <div class="sub" id="kpiUptime">Uptime: --s</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <span>Direct MJPEG Stream URL: <code>/video_feed</code></span>
            <span>JSON Telemetry API: <code>/stats</code></span>
        </div>
    </div>

    <script>
        async function updateTelemetry() {
            try {
                const res = await fetch('/stats');
                if (!res.ok) return;
                const data = await res.json();

                document.getElementById('kpiCount').innerText = data.people_count;
                document.getElementById('kpiDensity').innerHTML = data.density_per_megapixel + ' <span style="font-size: 1rem;">/ MP</span>';
                document.getElementById('kpiProb').innerText = (data.crowd_probability * 100).toFixed(1) + '%';
                document.getElementById('kpiFps').innerHTML = data.fps + ' <span style="font-size: 1rem;">FPS</span>';
                document.getElementById('kpiUptime').innerText = 'Uptime: ' + data.uptime_seconds + 's';

                const badge = document.getElementById('statusBadge');
                if (data.is_crowd) {
                    badge.className = 'status-pill status-crowd';
                    badge.innerText = 'CROWD ALERT';
                } else {
                    badge.className = 'status-pill status-normal';
                    badge.innerText = 'NORMAL';
                }
            } catch (e) {
                console.error("Telemetry update failed", e);
            }
        }
        setInterval(updateTelemetry, 500);
        updateTelemetry();
    </script>
</body>
</html>
"""


@app.get("/video_feed")
def video_feed():
    """Continuous Multipart MJPEG video stream accessible by browsers, VLC, or video tags."""
    def frame_generator():
        while True:
            if broadcaster is None:
                time.sleep(0.1)
                continue
            frame_bytes = broadcaster.get_latest_frame()
            if frame_bytes is None:
                time.sleep(0.02)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.01)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
def get_stats():
    """Live JSON telemetry state for automated alarm systems or web dashboards."""
    if broadcaster is None:
        return {"status": "INITIALIZING"}
    return broadcaster.get_stats()


@app.get("/health")
def healthcheck():
    """Standard container healthcheck."""
    return {
        "status": "healthy",
        "camera_connected": broadcaster.stats["camera_connected"] if broadcaster else False
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crowd Detection V1 — FastAPI Real-time Live Streaming Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source", default=os.getenv("CROWD_STREAM_SOURCE", "0"), help="Webcam index (0), RTSP URL, or video file path")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO IoU threshold")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (process every Nth frame, e.g. 2 for 50% compute savings)")
    parser.add_argument("--smooth", type=int, default=10, help="Temporal probability moving average window size")
    return parser.parse_args()


def main():
    global broadcaster
    args = parse_args()

    print()
    print("=" * 75)
    print(" CROWD DETECTION V1 — REAL-TIME FASTAPI STREAMING SERVER")
    print("=" * 75)
    print(f"Source         : {args.source}")
    print(f"Server Host    : http://{args.host}:{args.port}")
    print(f"Stride         : Process every {args.stride} frame(s)")
    print(f"Temporal Smooth: {args.smooth} frames")
    print(f"Dashboard URL  : http://localhost:{args.port}/")
    print(f"Live Stream    : http://localhost:{args.port}/video_feed")
    print(f"Telemetry API  : http://localhost:{args.port}/stats")
    print("=" * 75)
    print()

    # Initialize live broadcaster
    broadcaster = CrowdStreamBroadcaster(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        stride=args.stride,
        smooth=args.smooth
    )

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if broadcaster:
            broadcaster.stop()


if __name__ == "__main__":
    main()
