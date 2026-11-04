#!/usr/bin/env python3
# =============================================================================
# CROWD DETECTION V1
# Standalone Video Inference & Marking Script
#
# Usage:
#     python infer_video.py <video_path_or_cam_index> [options]
#
# Examples:
#     python infer_video.py input_crowd.mp4
#     python infer_video.py input_crowd.mp4 --output output/marked_crowd.mp4
#     python infer_video.py 0 --show                       # Live Webcam feed
#     python infer_video.py input.mp4 --stride 2           # Process every 2nd frame (15 FPS)
#     python infer_video.py rtsp://192.168.1.100:554/live   # IP Camera Stream
#
# Pipeline:
#     Video Stream (Webcam / File / RTSP)
#       ↓
#     YOLO26m Person Detector (CrowdHuman)
#       ↓
#     21 Scene-Level Geometric & Density Features
#       ↓
#     Random Forest Crowd Classifier
#       ↓
#     Temporal Moving-Average Filter (Flicker Prevention)
#       ↓
#     Frame Visual Annotation (Boxes + Analytical HUD Banner)
#       ↓
#     Marked Output Video (.mp4) / Live Display Window
# =============================================================================

import argparse
from collections import deque
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

# Suppress benign framework warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION & FEATURE COLUMNS
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
    """Find model weights across standard project directories."""
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

def extract_features(result) -> dict:
    """Extract the exact 21 spatial & density features used by Random Forest."""
    image = result.orig_img
    height, width = image.shape[:2]
    image_area = width * height

    # Fallback if zero detections
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

    # Normalized center points [0, 1]
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.column_stack([centers_x / width, centers_y / height])

    # Average center distance
    if person_count >= 2:
        pairwise_distances = []
        for i in range(person_count):
            for j in range(i + 1, person_count):
                pairwise_distances.append(np.linalg.norm(centers[i] - centers[j]))
        average_center_distance = float(np.mean(pairwise_distances))
    else:
        average_center_distance = 0.0

    # Average nearest neighbor distance
    if person_count >= 2:
        nearest_distances = []
        for i in range(person_count):
            distances = np.linalg.norm(centers - centers[i], axis=1)
            distances[i] = np.inf
            nearest_distances.append(np.min(distances))
        average_nearest_neighbor_distance = float(np.mean(nearest_distances))
    else:
        average_nearest_neighbor_distance = 0.0

    # 3x3 Spatial Grid
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
    """
    Renders high-visibility bounding boxes and a semi-transparent HUD banner.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Color palette (BGR)
    if is_crowd:
        theme_color = (34, 48, 220)       # Crimson / Vibrant Red
        badge_bg = (24, 30, 180)
        status_text = "CROWD ALERT"
    else:
        theme_color = (60, 180, 75)       # Emerald / Calming Green
        badge_bg = (40, 140, 55)
        status_text = "NORMAL DENSITY"

    # 1. Draw individual person bounding boxes
    if boxes is not None and len(boxes) > 0:
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Box outline
            cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_color, 2)

            # Tiny confidence tag
            tag = f"{conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(th + 4, y1)), theme_color, -1)
            cv2.putText(annotated, tag, (x1 + 2, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Semi-transparent Top HUD Banner
    hud_height = max(70, int(h * 0.10))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 18, 24), -1)
    # Blend with original frame for dark glassmorphism effect
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Accent bottom line on HUD
    cv2.line(annotated, (0, hud_height), (w, hud_height), theme_color, 2)

    # Scale factors based on frame width
    scale_title = max(0.65, min(1.0, w / 1280))
    scale_sub = max(0.45, min(0.70, w / 1280))

    # Status Badge
    badge_width = int(220 * scale_title)
    badge_height = int(hud_height * 0.55)
    cv2.rectangle(annotated, (14, 10), (14 + badge_width, 10 + badge_height), badge_bg, -1)
    cv2.rectangle(annotated, (14, 10), (14 + badge_width, 10 + badge_height), theme_color, 2)

    badge_label = f"[{status_text}]"
    cv2.putText(
        annotated,
        badge_label,
        (22, int(10 + badge_height * 0.70)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_title,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Probability Gauge Bar
    prob_bar_x = 22 + badge_width
    prob_bar_w = int(140 * scale_title)
    prob_bar_h = max(8, int(badge_height * 0.35))
    prob_bar_y = 10 + int(badge_height * 0.30)

    # Background bar
    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + prob_bar_w, prob_bar_y + prob_bar_h), (50, 50, 60), -1)
    # Filled probability
    fill_w = int(prob_bar_w * np.clip(crowd_prob, 0.0, 1.0))
    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + fill_w, prob_bar_y + prob_bar_h), theme_color, -1)
    cv2.rectangle(annotated, (prob_bar_x, prob_bar_y), (prob_bar_x + prob_bar_w, prob_bar_y + prob_bar_h), (120, 120, 130), 1)

    prob_text = f"{crowd_prob * 100:.1f}%"
    cv2.putText(
        annotated,
        prob_text,
        (prob_bar_x + prob_bar_w + 10, prob_bar_y + prob_bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_sub,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )

    # Analytical Metrics Sub-bar (Second Row in HUD)
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
# MAIN VIDEO INFERENCE PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crowd Detection V1 — Standalone Video & Live Stream Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source", nargs="?", default=None, help="Video file path, webcam index (0), or RTSP URL")
    parser.add_argument("-s", "--source-opt", dest="source_opt", default=None, help="Alternative flag for source")
    parser.add_argument("-o", "--output", default=None, help="Path for marked output video (.mp4)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=500, help="Max detections per frame")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (e.g. 2 or 3 for lower overhead)")
    parser.add_argument("--smooth", type=int, default=10, help="Temporal window size for moving average probability")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window during processing")
    parser.add_argument("--no-save", action="store_true", help="Skip saving annotated video to disk")

    args = parser.parse_args()

    # Determine input source
    source_val = args.source if args.source is not None else args.source_opt
    if source_val is None:
        print()
        print("=" * 75)
        print(" ❌ Error: No input video source provided.")
        print("=" * 75)
        print("Usage:")
        print("    python infer_video.py <video_path_or_cam_index> [options]")
        print()
        print("Examples:")
        print("    python infer_video.py input/sample_crowd.mp4")
        print("    python infer_video.py 0 --show              # Webcam")
        print("    python infer_video.py test.mp4 --stride 2   # Every 2nd frame")
        print("=" * 75)
        sys.exit(1)

    # Check if webcam index or file/stream
    if source_val.isdigit():
        capture_source = int(source_val)
        is_live_stream = True
        source_name = f"Webcam_{capture_source}"
    else:
        capture_source = source_val
        is_live_stream = source_val.startswith("rtsp://") or source_val.startswith("http://")
        source_name = Path(source_val).name

    # Check models
    yolo_path = resolve_model_path("best.pt")
    rf_path = resolve_model_path("crowd_random_forest_v1.pkl")

    if not yolo_path.exists():
        print(f"❌ YOLO model not found at: {yolo_path}")
        sys.exit(1)
    if not rf_path.exists():
        print(f"❌ Random Forest model not found at: {rf_path}")
        sys.exit(1)

    # Open video stream
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        print(f"❌ Could not open video source: {capture_source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live_stream else -1

    # Determine output path
    output_path = None
    writer = None
    if not args.no_save and not is_live_stream:
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        else:
            out_dir = SCRIPT_DIR / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(source_val).stem if not str(capture_source).isdigit() else "webcam"
            output_path = out_dir / f"{stem}_marked.mp4"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = src_fps / max(1, args.stride)
        writer = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (width, height))

    print()
    print("=" * 75)
    print(" CROWD DETECTION V1 — VIDEO INFERENCE PIPELINE")
    print("=" * 75)
    print(f"Source         : {source_name} ({width}x{height} @ {src_fps:.1f} FPS)")
    if total_frames > 0:
        print(f"Total Frames   : {total_frames}")
    print(f"Process Stride : Every {args.stride} frame(s)")
    print(f"Temporal Smooth: {args.smooth} frames buffer")
    print(f"YOLO Weights   : {yolo_path.name}")
    print(f"Classifier     : {rf_path.name}")
    if output_path:
        print(f"Output Video   : {output_path}")
    print("=" * 75)
    print()

    # Load models
    print("Loading models into memory...")
    yolo_model = YOLO(str(yolo_path))
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    rf_model.n_jobs = 1

    print("Pipeline ready. Starting inference loop...")
    print("-" * 75)

    # Inference state
    prob_history = deque(maxlen=max(1, args.smooth))
    frame_index = 0
    processed_count = 0
    crowd_frame_count = 0
    peak_people = 0
    start_time = time.time()
    last_frame_time = time.time()

    # Cached last prediction for skipped frames
    last_boxes = None
    last_confs = None
    last_is_crowd = False
    last_prob = 0.0
    last_features = {
        "person_count": 0,
        "people_per_megapixel": 0.0,
        "average_nearest_neighbor_distance": 0.0
    }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # Check stride
            if frame_index % args.stride != 0:
                continue

            processed_count += 1
            iter_start = time.time()

            # 1. YOLO Detection
            results = yolo_model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                classes=[0],
                max_det=args.max_det,
                verbose=False
            )
            result = results[0]

            # 2. Extract 21 Scene Features
            features = extract_features(result)
            peak_people = max(peak_people, features["person_count"])

            # 3. Random Forest Classification
            feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            raw_prob = float(rf_model.predict_proba(feature_df)[0][1])

            # 4. Temporal Smoothing Buffer
            prob_history.append(raw_prob)
            smoothed_prob = float(np.mean(prob_history))
            is_crowd = smoothed_prob >= 0.50

            if is_crowd:
                crowd_frame_count += 1

            # Get bounding boxes for annotation
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)

            # Compute Instantaneous FPS
            curr_time = time.time()
            fps_instant = 1.0 / max(curr_time - last_frame_time, 1e-4)
            last_frame_time = curr_time

            # 5. Draw Annotations & HUD Banner
            annotated_frame = draw_hud_and_boxes(
                frame=frame,
                boxes=boxes,
                confidences=confs,
                is_crowd=is_crowd,
                crowd_prob=smoothed_prob,
                person_count=features["person_count"],
                density_per_mp=features["people_per_megapixel"],
                nn_distance=features["average_nearest_neighbor_distance"],
                fps=fps_instant
            )

            # 6. Save or Display
            if writer is not None:
                writer.write(annotated_frame)

            if args.show:
                cv2.imshow("Crowd Detection V1 — Live Feed", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user ('q' key pressed).")
                    break

            # Progress log
            if processed_count % 10 == 0 or processed_count == 1:
                status_tag = "CROWD" if is_crowd else "NORMAL"
                progress_info = f"[{processed_count}/{total_frames}]" if total_frames > 0 else f"[Frame {processed_count}]"
                sys.stdout.write(
                    f"\r{progress_info} | FPS: {fps_instant:4.1f} | People: {features['person_count']:3d} | "
                    f"Prob: {smoothed_prob * 100:5.1f}% | Status: {status_tag}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time
    avg_fps = processed_count / max(total_elapsed, 1e-4)
    crowd_pct = (crowd_frame_count / max(processed_count, 1)) * 100

    print()
    print()
    print("=" * 75)
    print(" VIDEO PROCESSING SUMMARY")
    print("=" * 75)
    print(f"Total Processed Frames : {processed_count}")
    print(f"Elapsed Time           : {total_elapsed:.2f} seconds")
    print(f"Average Throughput     : {avg_fps:.1f} FPS")
    print(f"Peak People Count      : {peak_people}")
    print(f"Crowd State Ratio      : {crowd_pct:.1f}% of frames classified as CROWD")
    if output_path and output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Output Video File      : {output_path} ({size_mb:.2f} MB)")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
