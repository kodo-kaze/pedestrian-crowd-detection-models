# 🚶 Crowd Intelligence System: From Basic Classification to Predictive Risk Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Dataset: CrowdHuman](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CrowdHuman-orange.svg)](https://huggingface.co/datasets/sshao0516/CrowdHuman)
[![Paper: arXiv 1805.00123](https://img.shields.io/badge/Paper-arXiv%3A1805.00123-B31B1B.svg)](https://arxiv.org/abs/1805.00123)

---

## 📌 Project Overview

This repository houses a unified, multi-stage **Crowd Detection & Intelligence Framework**. The architecture evolves systematically across four progressive versions, moving from static single-frame crowd classification to dense spatial density analysis, real-time video multi-object tracking, and predictive crowd safety risk intelligence:

```text
V1: Basic Crowd Classification (CURRENT IMPLEMENTED RELEASE)
        ↓
V2: Density-Aware Crowd Detection (PLANNED)
        ↓
V3: Real-Time Crowd Tracking & Temporal Analysis (PLANNED)
        ↓
V4: Advanced Crowd Intelligence & Risk Detection (PLANNED)
```

Functional system pipeline:

```text
Image / Video Frame
        │
        ▼
┌──────────────┐
│    Person    │
│  Detection   │ ──► YOLO26m (CrowdHuman fine-tuned)
└───────┬──────┘
        │
        ▼
┌──────────────┐
│  Feature /   │
│   Density    │ ──► 21 Spatial, Geometric & Density Features
└───────┬──────┘
        │
        ▼
┌──────────────┐
│    Crowd     │
│  Classifier  │ ──► Random Forest Model (CROWD / NOT-CROWD)
└───────┬──────┘
        │
        ▼
┌──────────────┐
│  Tracking /  │
│   Temporal   │ ──► ByteTrack / BoT-SORT & dN/dt Analysis [V3]
└───────┬──────┘
        │
        ▼
┌──────────────┐
│ Risk & Flow  │
│ Intelligence │ ──► Congestion, Vector Conflicts & Risk Score [V4]
└──────────────┘
```

> 💡 **Implementation Status**: **V1 is fully implemented, trained, and tested in this repository.** It provides the core person detection, 21-feature extraction engine, trained Random Forest classifier, and the standalone [`infer.py`](file:///home/bimbok/shared/code/working_on/model/infer.py) inference script. Subsequent versions (V2, V3, V4) represent the forward-looking development roadmap detailed below.

---

## 📋 Table of Contents

1. [V1 — Basic Crowd Detection (Current Implementation)](#1-v1--basic-crowd-detection-current-implementation)
   - [Architecture & Dataflow](#architecture--dataflow)
   - [Stage 1: YOLO26m Person Detector](#stage-1-yolo26m-person-detector)
   - [Stage 2: The 21 Feature Extraction Engine](#stage-2-the-21-feature-extraction-engine)
   - [Stage 3: Random Forest Classifier & Benchmark Metrics](#stage-3-random-forest-classifier--benchmark-metrics)
   - [The V1 Operational Crowd Definition & Its Scope](#the-v1-operational-crowd-definition--its-scope)
   - [What V1 Can and Cannot Do](#what-v1-can-and-cannot-do)
2. [Standalone Inference with `infer.py`](#2-standalone-inference-with-inferpy)
   - [CLI Arguments & Usage](#cli-arguments--usage)
   - [Sample Execution Output](#sample-execution-output)
3. [The Foundation Dataset: CrowdHuman Deep Dive](#3-the-foundation-dataset-crowdhuman-deep-dive)
   - [Overview & Research Motivation](#overview--research-motivation)
   - [Dataset Scale & Key Statistics](#dataset-scale--key-statistics)
   - [Deep Benchmark Comparison](#deep-benchmark-comparison)
   - [Hugging Face Repository Layout & File Checksums](#hugging-face-repository-layout--file-checksums)
   - [The ODGT Annotation Protocol (fbox, vbox, hbox, mask)](#the-odgt-annotation-protocol)
   - [Downloading & Unzipping CrowdHuman](#downloading--unzipping-crowdhuman)
   - [Python ODGT Parser & Color Visualizer](#python-odgt-parser--color-visualizer)
   - [Format Conversion Pipelines (YOLO, COCO, VOC)](#format-conversion-pipelines)
   - [Evaluation Metrics (mMR, AP, Recall)](#evaluation-metrics)
   - [How `crowdhuman_ml_dataset.csv` Was Generated](#how-crowdhuman_ml_datasetcsv-was-generated)
4. [Evolutionary Roadmap: V2 → V3 → V4](#4-evolutionary-roadmap-v2--v3--v4)
   - [V2: Density-Aware Crowd Detection](#v2--density-aware-crowd-detection)
   - [V3: Real-Time Crowd Tracking & Temporal Analysis](#v3--real-time-crowd-tracking--temporal-analysis)
   - [V4: Advanced Crowd Intelligence & Risk Detection](#v4--advanced-crowd-intelligence--risk-detection)
   - [V1 → V4 Capabilities Comparison Matrix](#v1--v4-capabilities-comparison-matrix)
5. [Repository Structure](#5-repository-structure)
6. [Installation & Quickstart Guide](#6-installation--quickstart-guide)
7. [Citation & References](#7-citation--references)

---

## 1. V1 — Basic Crowd Detection (Current Implementation)

### Architecture & Dataflow

```text
                    INPUT IMAGE
                         │
                         ▼
                  ┌────────────┐
                  │  YOLO26m   │
                  │   Person   │
                  │ Detection  │
                  └─────┬──────┘
                        │
                        ▼
                Detected People
                        │
                        ▼
              Feature Extraction
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      Count         Density        Spatial
      Features      Features       Features
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                ┌──────────────┐
                │ Random Forest│
                └──────┬───────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
              CROWD      NOT-CROWD
```

---

### Stage 1: YOLO26m Person Detector

YOLO is strictly responsible for **individual person detection**:
- **Model Path**: `models/best.pt`
- **Trained on**: CrowdHuman dataset (dense human crowds, heavy mutual occlusion).
- **Inference Configuration**:
  - `conf = 0.25`: Balances high sensitivity with false-alarm suppression.
  - `iou = 0.70`: Permits adjacent overlapping pedestrians to survive suppression.
  - `max_det = 500`: Accommodates dense crowd scenes.
  - `classes = [0]`: Strictly isolates person bounding boxes.

```text
Input image
     ↓
YOLO26m
     ↓
┌───────────────────────┐
│ Person 1 → box + conf │
│ Person 2 → box + conf │
│ Person 3 → box + conf │
│ ...                   │
│ Person 50 → box + conf│
└───────────────────────┘
```

> ⚠️ **Key Architectural Distinction**: YOLO **never** decides whether an image is a crowd. It simply outputs: *"These are the individual people detected."* Deciding whether the scene constitutes a crowd is handled downstream by the 21-feature extraction engine and Random Forest classifier.

---

### Stage 2: The 21 Feature Extraction Engine

From YOLO's detections, 21 scene-level features are deterministically extracted:

#### 1. Basic Count (1 feature)
- `person_count`: Total number of detected people ($N$).

#### 2. Detection Confidence (3 features)
- `avg_confidence`: Mean detector confidence $\frac{1}{N} \sum c_i$.
- `min_confidence`: Minimum confidence score $\min(c_i)$.
- `max_confidence`: Maximum confidence score $\max(c_i)$.

#### 3. Bounding Box Geometry & Size (5 features)
- `total_bbox_area_ratio`: $\frac{\sum \text{area}_i}{\text{Image Area}}$ *(can exceed 1.0 due to mutual overlap)*.
- `avg_bbox_area_ratio`: Mean bounding box area relative to image area.
- `max_bbox_area_ratio`: Largest single bounding box area ratio (flags close-up foreground subjects).
- `avg_bbox_width_ratio`: Average bounding box width relative to image width.
- `avg_bbox_height_ratio`: Average bounding box height relative to image height.

#### 4. Density-Normalized (1 feature)
- `people_per_megapixel`:
  $$\text{people\_per\_megapixel} = \frac{\text{person\_count}}{\text{image\_width} \times \text{image\_height} / 1{,}000{,}000}$$
  Prevents high-resolution cameras from distorting count metrics.

#### 5. Spatial Arrangement (2 features)
- `average_center_distance`: Mean pairwise Euclidean distance across all detected person centers.
- `average_nearest_neighbor_distance`: Average distance from each person to their closest neighbor. Distinguishes dispersed people from tightly packed clusters:

```text
Dispersed (High Nearest-Neighbor Distance):
👤                  👤
        👤
                         👤

Tightly Packed (Low Nearest-Neighbor Distance):
👤👤👤
👤👤👤
👤👤👤
```

#### 6. 3×3 Spatial Grid Distribution (9 features)
The image is divided into a 3×3 grid, counting people in each cell:

```text
┌────────┬────────┬────────┐
│ grid00 │ grid01 │ grid02 │
├────────┼────────┼────────┤
│ grid10 │ grid11 │ grid12 │
├────────┼────────┼────────┤
│ grid20 │ grid21 │ grid22 │
└────────┴────────┴────────┘
```
Features: `grid_00`, `grid_01`, `grid_02`, `grid_10`, `grid_11`, `grid_12`, `grid_20`, `grid_21`, `grid_22`.

---

### Stage 3: Random Forest Classifier & Benchmark Metrics

- **Model File**: `models/crowd_random_forest_v1.pkl`
- **Inputs**: Exactly 21 features in the strict sequence defined above.
- **Output**: Binary classification (`CROWD` / `NOT-CROWD`) + calibrated probability score.

#### Performance on Held-Out Test Set:
- **Accuracy**: `91.53%`
- **Precision**: `90.26%`
- **Recall**: `88.73%`
- **F1-Score**: `89.49%`
- **ROC-AUC**: `0.9754`

---

### The V1 Operational Crowd Definition & Its Scope

In V1, the ground truth crowd status was established via an operational threshold:

```text
Ground-truth people >= 20  ──►  CROWD
Ground-truth people <  20  ──►  NOT-CROWD
```

> **Scientific Clarification**:
> A count of $\ge 20$ people is **not** an absolute universal law of crowds. It represents an **operational definition** selected to train an objective, reproducible baseline for V1.
> 
> **Limitations of V1**:
> - 15 people tightly packed in a small elevator is a dense crowd, but V1 rules it `NOT-CROWD` based purely on count.
> - 22 people scattered over a massive 100-meter football field is dispersed, but V1 rules it `CROWD`.
> - V1 cannot detect direction, movement, growth rate, bottlenecks, or panic.
> 
> **V2, V3, and V4 directly address these limitations.**

---

### What V1 Can and Cannot Do

| What V1 Can Do ✅ | What V1 Cannot Do ❌ (Deferred to V2/V3/V4) |
| :--- | :--- |
| Detect individuals in crowded environments | Assess whether a crowd is becoming hazardous |
| Extract 21 geometry, density, and spatial features | Measure real-time crowd growth rate ($dN/dt$) |
| Classify images into CROWD / NOT-CROWD | Track people over time or identify movement flow |
| Output confidence and class probabilities | Identify crowd bottlenecks or physical chokepoints |
| Export annotated visual detection images | Detect panic, stampedes, or counter-flow anomalies |

---

## 2. Standalone Inference with `infer.py`

Execute inference on any still image via the standalone script:

```bash
# Basic usage
python3 infer.py <path_to_image>

# Example
python3 infer.py input/station_platform.jpg
```

### Sample Execution Output:

```text
===========================================================================
 FINAL PREDICTION
===========================================================================

Image                 : station_platform.jpg
Detected people       : 55
Average confidence    : 0.7421
People / megapixel    : 48.12
Nearest-neighbor      : 0.0418

NOT-CROWD probability : 10.67%
CROWD probability     : 89.33%

Prediction            : CROWD

===========================================================================

✅ Annotated image saved:
   output/station_platform_prediction.jpg
===========================================================================
```

---

## 3. The Foundation Dataset: CrowdHuman Deep Dive

The entire training regime is backed by the [**CrowdHuman Dataset**](https://huggingface.co/datasets/sshao0516/CrowdHuman) (Shao et al., 2018 — Megvii Technology / Face++).

### Overview & Research Motivation

CrowdHuman was released to solve the severe data distribution mismatch between existing pedestrian benchmarks and real-world crowded environments:
- **Caltech Pedestrians & KITTI**: Restricted to automotive viewpoints, sparse pedestrian counts (0.3 to 0.8 persons/image), and mostly upright pedestrians.
- **CityPersons**: Focused on driving scenes with low overlap (pairwise IoU > 0.5 was only 0.02 pairs/image).
- **MS COCO**: Generic object detection with an average of only 3.82 persons per image.
- **CrowdHuman**: Scraped from Google Image Search across festivals, subways, markets, sports events, protests, and concerts. Provides **~470,000 human instances** across **24,370 images**, averaging **~22.64 persons per image**.

### Dataset Scale & Key Statistics

| Property | Training Set | Validation Set | Test Set | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Images** | 15,000 | 4,370 | 5,000 | **24,370** |
| **Annotated Human Instances** | 339,565 | 99,486 | *(Withheld for benchmarking)* | **~470,000** (train + val) |
| **Average Persons / Image** | 22.64 | 22.77 | ~22.6 | **~22.64** |
| **Min Persons in Single Image** | 1 | 1 | - | **1** |
| **Max Persons in Single Image** | 100+ | 100+ | - | **100+** |
| **Annotation Types per Person** | 3 (`fbox`, `vbox`, `hbox`) | 3 (`fbox`, `vbox`, `hbox`) | - | **3 bounding boxes** |
| **Total Boxes (Train + Val)** | ~1,018,695 | ~298,458 | - | **> 1.4 Million Bounding Boxes** |
| **Image Resolution Range** | ~600px to 6000px+ (Avg: ~1500 × 1000 px) | Varied aspect ratios | Varied aspect ratios | High-resolution web imagery |

### Deep Benchmark Comparison

| Dataset | Year | Environment / Viewpoint | Images (Train) | Instances (Train) | Persons / Image | Pairwise Overlap (IoU > 0.5) | Bounding Box Tiers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Caltech Pedestrians** | 2009 | Urban driving dashboard (640×480) | 42,782 frames | 13,674 | 0.32 | ~0.01 | Full, Visible |
| **KITTI** | 2012 | Roof-mounted camera (1242×375) | 7,481 | 4,487 | 0.84 | ~0.02 | 2D box, 3D box |
| **CityPersons** | 2017 | Urban driving (2048×1024) | 2,975 | 19,650 | 6.47 | 0.02 | Full, Visible |
| **MS COCO (Person class)** | 2014 | Web images / Everyday scenes | 64,115 | 273,469 | 3.82 | 0.11 | Full body |
| **WiderPerson** | 2019 | Web images / Dense crowds | 9,000 | 255,238 | 28.36 | 1.82 | Full body, 5 sub-classes |
| **CrowdHuman (Ours)** | **2018** | **Universal Web Search / Any Scene** | **15,000** | **339,565** | **22.64** | **2.40** | **Full (`fbox`), Visible (`vbox`), Head (`hbox`)** |

---

### Hugging Face Repository Layout & File Checksums

Repository: [`sshao0516/CrowdHuman`](https://huggingface.co/datasets/sshao0516/CrowdHuman) (Total download: **~14.2 GB**; uncompressed: **~16.5 GB**):

| File Name | Exact Size (Bytes) | Human Size | Git LFS SHA-256 Hash | Description |
| :--- | :--- | :--- | :--- | :--- |
| `CrowdHuman_train01.zip` | 2,970,597,373 | 2.97 GB | `7ba340163cff0f2446027af95dc96bcb9c66be18506eabb57822c400a7efd3b8` | Train images Part 1 |
| `CrowdHuman_train02.zip` | 3,092,749,718 | 3.09 GB | `d9ecfb43eaf8381ddd4d1ff4e9a0877b694dbfba7a7a33ed3966ee2c2a628663` | Train images Part 2 |
| `CrowdHuman_train03.zip` | 2,306,357,030 | 2.31 GB | `9cac171914c4f5c7371e3d42d20b50858caa1602811ae573aa35bbfe672dc29f` | Train images Part 3 |
| `CrowdHuman_val.zip` | 2,488,658,160 | 2.49 GB | `c0ab99bb80ac162cd3efdf94a1a6100c4f059a61d69596412c6b44ebc20d1363` | Validation images (4,370 files) |
| `CrowdHuman_test.zip` | 3,259,241,265 | 3.26 GB | `4d43213edab47ca26d36bd1a8438f859da45b74e160c4d147871426d95914213` | Test images (5,000 files) |
| `annotation_train.odgt` | 80,017,502 | 80.0 MB | `6bf241a79f19e30cf52681eab3392368bd5a534164be9272e7a808cb284d9f77` | 15,000 JSON lines |
| `annotation_val.odgt` | 23,323,139 | 23.3 MB | `be422c79a190ff7e30fe5cbd74cbf45a2dadda6c5af58c6ec11a038ba2993c04` | 4,370 JSON lines |

> **Note on `datasets.load_dataset()`**: Attempting to use the Hugging Face `datasets` library directly triggers a `DatasetGenerationError` because the automated dataset viewer expects Parquet tables, whereas the repository contains raw `.zip` archives and `.odgt` text files. Download using `huggingface_hub` or CLI instead.

---

### The ODGT Annotation Protocol

All coordinates use $[x, y, w, h]$ pixel values:
- `fbox`: Full body box (inferred/extrapolated through occlusions).
- `vbox`: Visible body box (only unoccluded pixels).
- `hbox`: Head box.
- `tag = "person"`: Standard human target.
- `tag = "mask"`: Ignored region (background crowds, mannequins, silhouettes, reflections).
- `extra.ignore = 1`: Skip in loss function and metric evaluations.

```json
{
  "ID": "273271,104ec00067d5b782",
  "gtboxes": [
    {
      "tag": "person",
      "fbox": [419, 291, 149, 381],
      "vbox": [419, 291, 149, 381],
      "hbox": [471, 291, 62, 78],
      "extra": { "box_id": 0, "occ": 0, "ignore": 0 },
      "head_attr": { "occ": 0, "unsure": 0, "ignore": 0 }
    },
    {
      "tag": "mask",
      "fbox": [0, 210, 110, 180],
      "vbox": [0, 210, 110, 180],
      "hbox": [0, 0, 0, 0],
      "extra": { "ignore": 1 }
    }
  ]
}
```

---

### Downloading & Unzipping CrowdHuman

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sshao0516/CrowdHuman",
    repo_type="dataset",
    local_dir="./CrowdHuman",
    local_dir_use_symlinks=False,
    resume_download=True
)
```

Unzipping and merging the split train archives:

```bash
cd ./CrowdHuman
mkdir -p images/train images/val images/test

# Extract validation set
unzip -q CrowdHuman_val.zip -d val_temp/
find val_temp/ -type f \( -name "*.jpg" -o -name "*.png" \) -exec mv {} images/val/ \;
rm -rf val_temp/

# Extract and merge the 3 training archives
unzip -q CrowdHuman_train01.zip -d train_temp/
unzip -q CrowdHuman_train02.zip -d train_temp/
unzip -q CrowdHuman_train03.zip -d train_temp/
find train_temp/ -type f \( -name "*.jpg" -o -name "*.png" \) -exec mv {} images/train/ \;
rm -rf train_temp/
```

---

### Python ODGT Parser & Color Visualizer

```python
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def parse_odgt(odgt_path: Path):
    """Generator yielding parsed records from an ODGT file."""
    with open(odgt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line.strip())

def visualize_sample(image_path: Path, annotation_record: dict, output_file: Path = None):
    """Renders boxes: Full Body (Blue), Visible Body (Green), Head (Red), Mask (Gray)."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in annotation_record.get("gtboxes", []):
        tag = box.get("tag", "person")
        is_ignore = box.get("extra", {}).get("ignore", 0) == 1

        if tag == "mask" or is_ignore:
            x, y, w, h = [int(v) for v in box["fbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (128, 128, 128), 2)
            continue

        # 1. Full body (fbox) -> Blue
        if "fbox" in box:
            fx, fy, fw, fh = [int(v) for v in box["fbox"]]
            cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 102, 255), 2)

        # 2. Visible body (vbox) -> Green
        if "vbox" in box:
            vx, vy, vw, vh = [int(v) for v in box["vbox"]]
            cv2.rectangle(image, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 1)

        # 3. Head (hbox) -> Red
        if "hbox" in box:
            hx, hy, hw, hh = [int(v) for v in box["hbox"]]
            cv2.rectangle(image, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)

    plt.figure(figsize=(12, 8), dpi=150)
    plt.imshow(image)
    plt.axis("off")
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
```

---

### Format Conversion Pipelines

#### Convert ODGT to YOLO Format:
```python
import json
from pathlib import Path
import cv2
from tqdm import tqdm

def convert_odgt_to_yolo(odgt_path: Path, images_dir: Path, output_labels_dir: Path, box_key: str = "fbox"):
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    img_map = {p.stem: p for p in images_dir.rglob("*.jpg")}

    with open(odgt_path, "r") as f:
        for line in tqdm(f, desc="Converting to YOLO"):
            rec = json.loads(line.strip())
            img_path = img_map.get(rec["ID"])
            if not img_path:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            labels = []
            for b in rec.get("gtboxes", []):
                if b.get("tag") != "person" or b.get("extra", {}).get("ignore", 0) == 1:
                    continue
                bbox = b.get(box_key)
                if not bbox:
                    continue
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                cx = min(max((x + w / 2.0) / W, 0.0), 1.0)
                cy = min(max((y + h / 2.0) / H, 0.0), 1.0)
                nw = min(max(w / W, 0.0), 1.0)
                nh = min(max(h / H, 0.0), 1.0)
                labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(output_labels_dir / f"{rec['ID']}.txt", "w") as out:
                out.write("\n".join(labels))
```

---

### Evaluation Metrics

- **mMR (log-average Miss Rate)**: Evaluates Miss Rate vs False Positives Per Image (FPPI) in the range $[10^{-2}, 10^{0}]$. Lower is better ($0\%$ is optimal).
- **AP (Average Precision @ IoU=0.5)**: Standard VOC/COCO detection precision. Higher is better ($100\%$ is optimal).
- **Recall**: Proportion of true targets successfully detected.

#### Official Baseline (ResNet-50-FPN Faster R-CNN on CrowdHuman Val):
- **Full Body (`fbox`)**: AP = 85.0%, mMR = 49.7%, Recall = 91.2%
- **Visible Body (`vbox`)**: AP = 79.8%, mMR = 55.9%, Recall = 86.6%
- **Head (`hbox`)**: AP = 70.4%, mMR = 53.1%, Recall = 76.5%

---

### How `crowdhuman_ml_dataset.csv` Was Generated

Located at [`datasets/crowdhuman_ml_dataset.csv`](file:///home/bimbok/shared/code/working_on/model/datasets/crowdhuman_ml_dataset.csv), this 4,371-row dataset was engineered by:
1. Iterating through all 4,370 validation images from CrowdHuman.
2. Running YOLO person detection on each image.
3. Extracting the 21 geometric, density, and spatial grid features.
4. Parsing ground truth counts from `annotation_val.odgt`.
5. Setting target label `1` if $\text{count} \ge 20$, else `0`.
6. Training and validating the Random Forest classifier.

---

## 4. Evolutionary Roadmap: V2 → V3 → V4

### 🔵 V2 — Density-Aware Crowd Detection

**Goal**: Move beyond raw person counts. Incorporate scene context, density, and space occupancy.

```text
                    IMAGE
                      │
                      ▼
                 YOLO26m
                      │
                      ▼
              Person detections
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Count       Density      Spatial
                    Map          Analysis
          │           │           │
          └───────────┼───────────┘
                      ▼
              Crowd Classifier
                      │
             ┌────────┴────────┐
             ▼                 ▼
          CROWD           NOT-CROWD
```

#### Planned Innovations:
1. **Scene-Level Human Labels**: Replaces arbitrary count rules with qualitative crowd annotations.
2. **Spatial Density Heatmaps**:
   ```text
   ┌────────────────────────┐
   │                        │
   │       ░░░░░            │
   │      ░████░            │  ◄── Dense Local Hotspot
   │      ░████░            │
   │                        │
   │                ░       │
   └────────────────────────┘
   ```
3. **Local vs Global Density**: Flags localized cluster hazards even if the broader space is empty.
4. **Area Occupancy Estimation**: Measures the proportion of traversable area occupied by people.

---

### 🟠 V3 — Real-Time Crowd Tracking & Temporal Analysis

**Goal**: Introduce the **time dimension** ($t$) through multi-frame video stream analysis.

```text
                     VIDEO
                       │
                       ▼
                  Frame Stream
                       │
                       ▼
                    YOLO26m
                       │
                       ▼
                Person Detection
                       │
                       ▼
                Object Tracking
                       │
                 ┌─────┴─────┐
                 ▼           ▼
              ByteTrack   BoT-SORT
                 │           │
                 └─────┬─────┘
                       ▼
                Track Histories
                       │
                       ▼
              Temporal Analysis
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Density       Flow       Growth Rate
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Crowd Status
```

#### Planned Innovations:
1. **Multi-Object Tracking (ByteTrack / BoT-SORT)**: Assigns persistent IDs across frames.
2. **Crowd Growth Rate ($dN/dt$)**: Detects rapid crowd surges or sudden influxes.
3. **Motion Flow Vectors**: Computes velocity, direction, and speed consistency.
4. **Convergence Detection**: Identifies whether crowd vectors converge toward a common bottleneck or exit.

---

### 🔴 V4 — Advanced Crowd Intelligence & Risk Detection

**Goal**: Enterprise-grade crowd intelligence and automated predictive risk alerting.

```text
                         VIDEO
                           │
                           ▼
                       YOLO26m
                           │
                           ▼
                       Tracking
                           │
                           ▼
                  Density Estimation
                           │
                           ▼
                  Spatial Analysis
                           │
                           ▼
                  Temporal Analysis
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
        Crowd Size      Flow          Density
                         │
            ┌────────────┼──────────────┐
            ▼            ▼              ▼
        Congestion   Direction     Acceleration
            │            │              │
            └────────────┼──────────────┘
                         ▼
                   Risk Analysis
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         Normal       Warning       Critical
```

#### Planned Innovations:
1. **Dedicated Density Estimation**: Integrates crowd counting models for extreme occlusions where bounding boxes fail.
2. **Congestion Detection**: Detects high-density clusters combined with zero forward velocity.
3. **Flow Conflict & Turbulence**: Identifies counter-directional currents and abrupt dispersal anomalies.
4. **Calibrated Risk Engine (0-100)**: Translates multi-sensor dynamics into actionable safety alerts:
   - `NORMAL (0 - 45)`
   - `WARNING (46 - 75)`
   - `CRITICAL (76 - 100)`

---

### V1 → V4 Capabilities Comparison Matrix

| Capability | V1 (Current) | V2 (Planned) | V3 (Planned) | V4 (Planned) |
| :--- | :---: | :---: | :---: | :---: |
| **Person Detection** | ✅ YOLO26m | ✅ YOLO26m | ✅ YOLO26m | ✅ YOLO26m + Density Nets |
| **Crowd Classification** | ✅ (Count-based RF) | ✅ (Scene-based RF) | ✅ (Temporal RF) | ✅ (Multi-modal Engine) |
| **Spatial Feature Analysis** | ✅ (21 features) | ✅ (Advanced) | ✅ (Advanced) | ✅ (Full Spatial Field) |
| **Density Analysis** | ⚠️ Basic (Count/MP) | ✅ **Density Maps** | ✅ Advanced | ✅ Advanced |
| **Area Occupancy Ratio** | ❌ | ✅ **Planned** | ✅ Planned | ✅ Planned |
| **Multi-Object Tracking** | ❌ | ❌ | ✅ **ByteTrack / BoT-SORT** | ✅ Included |
| **Crowd Growth Rate ($dN/dt$)**| ❌ | ❌ | ✅ **Planned** | ✅ Included |
| **Motion Flow Vectors** | ❌ | ❌ | ✅ **Planned** | ✅ Included |
| **Convergence / Chokepoints**| ❌ | ❌ | ✅ **Planned** | ✅ Included |
| **Dedicated Counting Models**| ❌ | Optional | Optional | ✅ **Planned** |
| **Congestion Detection** | ❌ | Basic | Advanced | ✅ **Planned** |
| **Risk Scoring Engine (0-100)**| ❌ | ❌ | Basic Trend | ✅ **Full Engine** |
| **Automated Alerts** | ❌ | Optional | ✅ Planned | ✅ **Planned** |

---

## 5. Repository Structure

```
.
├── README.md                           # Unified Master Technical Documentation (This file)
├── requirements.txt                    # Core Python dependencies
├── infer.py                            # V1 Standalone image inference executable
├── infer_video.py                      # V1 Standalone video, webcam & RTSP inference executable
├── stream_server.py                    # V1 Real-Time FastAPI 24/7 CCTV Streaming Server
│
├── models/
│   ├── best.pt                         # Fine-tuned YOLO26m weights (~131 MB)
│   ├── crowd_random_forest_v1.pkl      # Trained 21-feature Random Forest model (~10 MB)
│   └── model.txt                       # Cloud backup storage mirrors
│
├── datasets/
│   └── crowdhuman_ml_dataset.csv       # Extracted 21-feature validation dataset (4,371 rows)
│
├── yolo26m_fine_tune_ml_train_code_v1/ # Jupyter Notebooks for training & validation
│   ├── pedestrian_0_yolo_ft-v1.ipynb   # YOLO conversion, data verification & fine-tuning
│   ├── pedestrian_1_yolo_ft-v1.ipynb   # Model validation & benchmark scoring
│   └── pedestrian_ML_train.ipynb       # Feature engineering & Random Forest training
│
├── input/                              # User input images directory
└── output/                             # Annotated inference outputs directory
```

---

## 6. Installation & Quickstart Guide

### Step 1: Environment Setup
```bash
git clone <your-repo-url>
cd model

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Image Inference
```bash
python3 infer.py input/your_test_image.jpg
```
Predictions and probability breakdowns will display in the console, with annotated images saved automatically to `output/`.

### Step 4: Run Video & File Processing
```bash
# 1. Process a video file (outputs marked MP4 to output/<name>_marked.mp4)
python3 infer_video.py input/your_video.mp4

# 2. Process with 50% compute savings (skip every 2nd frame)
python3 infer_video.py input/your_video.mp4 --stride 2

# 3. Real-time webcam feed with live desktop window
python3 infer_video.py 0 --show
```

### Step 5: Launch 24/7 FastAPI CCTV Streaming Server
Start the real-time background streaming server to ingest continuous camera feeds and broadcast modified live video over the web:

```bash
# 1. Stream from a live CCTV / IP Camera RTSP feed
python3 stream_server.py --source rtsp://username:password@192.168.1.100:554/live

# 2. Stream from local webcam
python3 stream_server.py --source 0

# 3. Stream from a test video in an infinite loop
python3 stream_server.py --source input/12781687_1920_1080_30fps.mp4 --stride 2
```

Once running, access:
* **Live Web Dashboard**: [http://localhost:8000/](http://localhost:8000/) (Real-time marked video feed + live KPI telemetry cards)
* **Direct Video Stream (MJPEG)**: [http://localhost:8000/video_feed](http://localhost:8000/video_feed) (Embeddable in `<img src="/video_feed">` or VLC)
* **Real-time Telemetry JSON API**: [http://localhost:8000/stats](http://localhost:8000/stats)
* **Healthcheck**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 7. Citation & References

```bibtex
@article{shao2018crowdhuman,
  title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
  author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:1805.00123},
  year={2018}
}

@software{Jocher_Ultralytics_YOLO_2024,
  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title = {{Ultralytics YOLO}},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}
```
