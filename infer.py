#!/usr/bin/env python3

# =============================================================================
# CROWD DETECTION V1
# Standalone Inference Script
#
# Usage:
#     python infer.py <image_path>
#
# Required files:
#     infer.py
#     best.pt
#     crowd_random_forest_v1.pkl
#
# Pipeline:
#
#     Image
#       ↓
#     YOLO26m
#       ↓
#     Person detections
#       ↓
#     21 scene-level features
#       ↓
#     Random Forest
#       ↓
#     CROWD / NOT-CROWD
# =============================================================================

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

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

YOLO_MODEL_PATH = resolve_model_path("best.pt")
RF_MODEL_PATH = resolve_model_path("crowd_random_forest_v1.pkl")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 500


# =============================================================================
# FEATURE COLUMNS
#
# IMPORTANT:
# These MUST remain in exactly the same order as used during training.
# =============================================================================

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

    "grid_00",
    "grid_01",
    "grid_02",

    "grid_10",
    "grid_11",
    "grid_12",

    "grid_20",
    "grid_21",
    "grid_22",
]


# =============================================================================
# ARGUMENT CHECK
# =============================================================================

if len(sys.argv) != 2:

    print()
    print("Usage:")
    print("    python infer.py <image_path>")
    print()
    print("Example:")
    print("    python infer.py test.jpg")
    print()

    sys.exit(1)


IMAGE_PATH = Path(sys.argv[1]).expanduser().resolve()


# =============================================================================
# FILE CHECKS
# =============================================================================

if not IMAGE_PATH.exists():

    print(f"❌ Image not found:")
    print(f"   {IMAGE_PATH}")

    sys.exit(1)


if not YOLO_MODEL_PATH.exists():

    print(f"❌ YOLO model not found:")
    print(f"   {YOLO_MODEL_PATH}")

    sys.exit(1)


if not RF_MODEL_PATH.exists():

    print(f"❌ Random Forest model not found:")
    print(f"   {RF_MODEL_PATH}")

    sys.exit(1)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(result):
    """
    Extract the exact 21 features used by the V1 Random Forest.

    Input:
        Ultralytics YOLO result for one image.

    Output:
        Dictionary containing all 21 features.
    """

    image = result.orig_img

    height, width = image.shape[:2]

    image_area = width * height


    # -------------------------------------------------------------------------
    # No detections
    # -------------------------------------------------------------------------

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

            "grid_00": 0,
            "grid_01": 0,
            "grid_02": 0,

            "grid_10": 0,
            "grid_11": 0,
            "grid_12": 0,

            "grid_20": 0,
            "grid_21": 0,
            "grid_22": 0,
        }


    # -------------------------------------------------------------------------
    # YOLO detections
    # -------------------------------------------------------------------------

    boxes = result.boxes.xyxy.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()


    # -------------------------------------------------------------------------
    # Person count
    # -------------------------------------------------------------------------

    person_count = len(boxes)


    # -------------------------------------------------------------------------
    # Confidence statistics
    # -------------------------------------------------------------------------

    avg_confidence = float(
        np.mean(confidences)
    )

    min_confidence = float(
        np.min(confidences)
    )

    max_confidence = float(
        np.max(confidences)
    )


    # -------------------------------------------------------------------------
    # Bounding-box dimensions
    # -------------------------------------------------------------------------

    bbox_widths = (
        boxes[:, 2] - boxes[:, 0]
    )

    bbox_heights = (
        boxes[:, 3] - boxes[:, 1]
    )

    bbox_areas = (
        bbox_widths * bbox_heights
    )


    # -------------------------------------------------------------------------
    # Bounding-box area ratios
    # -------------------------------------------------------------------------

    bbox_area_ratios = (
        bbox_areas / image_area
    )

    total_bbox_area_ratio = float(
        np.sum(bbox_area_ratios)
    )

    avg_bbox_area_ratio = float(
        np.mean(bbox_area_ratios)
    )

    max_bbox_area_ratio = float(
        np.max(bbox_area_ratios)
    )


    # -------------------------------------------------------------------------
    # Bounding-box width / height ratios
    # -------------------------------------------------------------------------

    avg_bbox_width_ratio = float(
        np.mean(bbox_widths / width)
    )

    avg_bbox_height_ratio = float(
        np.mean(bbox_heights / height)
    )


    # -------------------------------------------------------------------------
    # People per megapixel
    # -------------------------------------------------------------------------

    image_megapixels = (
        image_area / 1_000_000
    )

    people_per_megapixel = float(
        person_count / image_megapixels
    )


    # -------------------------------------------------------------------------
    # Bounding-box centers
    #
    # Normalize coordinates to:
    #
    #     x = 0 → left
    #     x = 1 → right
    #     y = 0 → top
    #     y = 1 → bottom
    # -------------------------------------------------------------------------

    centers_x = (
        boxes[:, 0] + boxes[:, 2]
    ) / 2

    centers_y = (
        boxes[:, 1] + boxes[:, 3]
    ) / 2

    centers = np.column_stack([
        centers_x / width,
        centers_y / height
    ])


    # -------------------------------------------------------------------------
    # Average center distance
    # -------------------------------------------------------------------------

    if person_count >= 2:

        pairwise_distances = []

        for i in range(person_count):

            for j in range(i + 1, person_count):

                distance = np.linalg.norm(
                    centers[i] - centers[j]
                )

                pairwise_distances.append(
                    distance
                )

        average_center_distance = float(
            np.mean(pairwise_distances)
        )

    else:

        average_center_distance = 0.0


    # -------------------------------------------------------------------------
    # Average nearest-neighbor distance
    # -------------------------------------------------------------------------

    if person_count >= 2:

        nearest_distances = []

        for i in range(person_count):

            distances = np.linalg.norm(
                centers - centers[i],
                axis=1
            )

            distances[i] = np.inf

            nearest_distances.append(
                np.min(distances)
            )

        average_nearest_neighbor_distance = float(
            np.mean(nearest_distances)
        )

    else:

        average_nearest_neighbor_distance = 0.0


    # -------------------------------------------------------------------------
    # 3 × 3 spatial grid
    #
    #        column
    #       0   1   2
    #
    # row 0 |   |   |
    # row 1 |   |   |
    # row 2 |   |   |
    # -------------------------------------------------------------------------

    grid_counts = np.zeros(
        (3, 3),
        dtype=int
    )

    for cx, cy in centers:

        col = min(
            int(cx * 3),
            2
        )

        row = min(
            int(cy * 3),
            2
        )

        grid_counts[row, col] += 1


    # -------------------------------------------------------------------------
    # Return all 21 features
    # -------------------------------------------------------------------------

    return {

        "person_count":
            person_count,

        "avg_confidence":
            avg_confidence,

        "min_confidence":
            min_confidence,

        "max_confidence":
            max_confidence,

        "total_bbox_area_ratio":
            total_bbox_area_ratio,

        "avg_bbox_area_ratio":
            avg_bbox_area_ratio,

        "max_bbox_area_ratio":
            max_bbox_area_ratio,

        "avg_bbox_width_ratio":
            avg_bbox_width_ratio,

        "avg_bbox_height_ratio":
            avg_bbox_height_ratio,

        "people_per_megapixel":
            people_per_megapixel,

        "average_center_distance":
            average_center_distance,

        "average_nearest_neighbor_distance":
            average_nearest_neighbor_distance,

        "grid_00":
            int(grid_counts[0, 0]),

        "grid_01":
            int(grid_counts[0, 1]),

        "grid_02":
            int(grid_counts[0, 2]),

        "grid_10":
            int(grid_counts[1, 0]),

        "grid_11":
            int(grid_counts[1, 1]),

        "grid_12":
            int(grid_counts[1, 2]),

        "grid_20":
            int(grid_counts[2, 0]),

        "grid_21":
            int(grid_counts[2, 1]),

        "grid_22":
            int(grid_counts[2, 2]),
    }


# =============================================================================
# LOAD MODELS
# =============================================================================

print()
print("=" * 75)
print(" CROWD DETECTION V1 — IMAGE INFERENCE")
print("=" * 75)

print()
print("Loading models...")

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)

with open(
    RF_MODEL_PATH,
    "rb"
) as f:

    rf_model = pickle.load(f)
rf_model.n_jobs = 1

print("✅ YOLO26m loaded")
print("✅ Random Forest loaded")


# =============================================================================
# RUN YOLO
# =============================================================================

print()
print("Running YOLO26m...")
print("-" * 75)

results = yolo_model.predict(
    source=str(IMAGE_PATH),
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    max_det=MAX_DETECTIONS,
    verbose=False
)

result = results[0]

print("✅ YOLO inference complete")


# =============================================================================
# EXTRACT FEATURES
# =============================================================================

print()
print("Extracting scene features...")
print("-" * 75)

features = extract_features(result)

print("✅ 21 features extracted")


# =============================================================================
# CREATE MODEL INPUT
# =============================================================================

feature_df = pd.DataFrame(
    [features],
    columns=FEATURE_COLUMNS
)


# =============================================================================
# RANDOM FOREST PREDICTION
# =============================================================================

prediction = int(
    rf_model.predict(feature_df)[0]
)

probabilities = (
    rf_model.predict_proba(feature_df)[0]
)

not_crowd_probability = float(
    probabilities[0]
)

crowd_probability = float(
    probabilities[1]
)


if prediction == 1:

    final_prediction = "CROWD"

else:

    final_prediction = "NOT-CROWD"


# =============================================================================
# FINAL RESULT
# =============================================================================

print()
print("=" * 75)
print(" FINAL PREDICTION")
print("=" * 75)

print()
print(f"Image                 : {IMAGE_PATH.name}")

print(
    f"Detected people      : "
    f"{features['person_count']}"
)

print(
    f"Average confidence   : "
    f"{features['avg_confidence']:.4f}"
)

print(
    f"People / megapixel   : "
    f"{features['people_per_megapixel']:.2f}"
)

print(
    f"Nearest-neighbor     : "
    f"{features['average_nearest_neighbor_distance']:.4f}"
)

print()

print(
    f"NOT-CROWD probability: "
    f"{not_crowd_probability * 100:.2f}%"
)

print(
    f"CROWD probability    : "
    f"{crowd_probability * 100:.2f}%"
)

print()

print(
    f"Prediction            : "
    f"{final_prediction}"
)

print()
print("=" * 75)


# =============================================================================
# OPTIONAL: SAVE YOLO VISUALIZATION
# =============================================================================


OUTPUT_DIR = SCRIPT_DIR / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR /
    f"{IMAGE_PATH.stem}_prediction.jpg"
)

annotated = result.plot()

cv2.imwrite(
    str(output_path),
    annotated
)

print()
print("✅ Annotated image saved:")
print(f"   {output_path}")

print()
print("=" * 75)
