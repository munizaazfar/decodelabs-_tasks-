# Project 4: Real-Time Object Detection Pipeline (MobileNet-SSD)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision pipeline that performs object detection and spatial location bounding using a pre-trained **MobileNet-SSD (Single Shot MultiBox Detector)** architecture integrated with OpenCV's `cv2.dnn` module.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture & Mathematical Model](#technical-architecture--mathematical-model)
- [Directory Structure](#directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Execution Workflow](#execution-workflow)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Outputs & Visual Results](#outputs--visual-results)
- [Verification & Quality Gatekeeper](#verification--quality-gatekeeper)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview
This repository implements an automated object detection pipeline capable of identifying up to 20 distinct object classes (derived from the PASCAL VOC dataset) plus background. Designed as part of the **DecodeLabs Architect's Playbook**, this project demonstrates end-to-end computer vision operations: image ingestion, multi-dimensional blob pre-processing, forward pass inference, confidence filtering, coordinate mapping, and visual annotation.

---

## ✨ Key Features
* **Pre-trained Caffe Integration**: Leverages `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` for fast, low-latency inference.
* **Automated Weight Downloader**: Includes `download_weights.py` to seamlessly fetch network architecture and model weights.
* **4D Blob Construction**: Standardizes input matrices ($300 \times 300$) with mean subtraction and scaling factors.
* **Configurable Confidence Gate**: Enforces a configurable confidence threshold (e.g., $80\%$ or $50\%$) to eliminate false positives.
* **Spatial Coordinate Mapping**: Maps normalized neural network predictions $[0, 1]$ directly back to pixel dimensions.
* **Visual Bounding Overlays**: Draws color-coded bounding boxes along with class labels and certainty percentage indicators.

---

## 🧠 Technical Architecture & Mathematical Model

### 1. Ingestion & Pre-Processing (4D Blob Construction)
The input matrix $I \in \mathbb{R}^{H \times W \times 3}$ is rescaled to a uniform $300 \times 300$ resolution. Mean subtraction ($127.5$) and scaling factor ($\alpha = 0.007843 \approx \frac{1}{127.5}$) are applied:
$$\text{Blob}(x, y) = (I(x, y) - 127.5) \times 0.007843$$

### 2. Forward Inference & Bounding Box Scaling
The network predicts normalized bounding box coordinates $[x_{\text{min}}, y_{	ext{min}}, x_{	ext{max}}, y_{	ext{max}}] \in [0, 1]$. These are transformed into pixel coordinates:
$$X_{\text{start}} = x_{\text{min}} \times W, \quad Y_{\text{start}} = y_{\text{min}} \times H$$
$$X_{\text{end}} = x_{\text{max}} \times W, \quad Y_{\text{end}} = y_{\text{max}} \times H$$

### 3. Confidence Gatekeeper
Detections are filtered based on predicted class confidence score $C_i$:
$$\text{Keep Detection} \iff C_i \ge \text{Confidence Gate}$$

---

## 📂 Directory Structure

```
Project4/
├── download_weights.py           # Helper script to download Caffe model files
├── main.py                       # Main pipeline execution script
├── MobileNetSSD_deploy.prototxt  # Caffe network architecture configuration
├── MobileNetSSD_deploy.caffemodel # Pre-trained neural network weights
├── test_image.jpg                # Sample input image
├── output_detection_result.png   # Generated output with overlaid bounding boxes
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

---

## ⚙️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/object-detection-mobilenet.git
cd object-detection-mobilenet/Project4
```

### 2. Install Required Dependencies
Ensure you have Python 3.8+ installed. Install OpenCV and numpy:
```bash
pip install opencv-python==4.10.0.84 numpy requests
```

---

## 🚀 Execution Workflow

### Step 1: Download Caffe Weights
Run the downloader script to fetch required architecture and weight files:
```bash
python download_weights.py
```

### Step 2: Place Input Image
Place your target JPEG/PNG image inside the `Project4` folder and name it `test_image.jpg`.

### Step 3: Run Object Detection
Execute the main pipeline:
```bash
python main.py
```

---

## 🎛️ Configuration & Hyperparameters

You can adjust the confidence gate threshold inside `main.py`:
```python
# For strict 80% confidence filtering:
run_object_detection(confidence_gate=0.80)

# For broader detection sensitivity:
run_object_detection(confidence_gate=0.50)
```

### Supported Object Classes (PASCAL VOC 20 Classes):
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

---

## 📊 Visual Results

Upon execution, the pipeline produces console logs and exports `output_detection_result.png`:

**Console Output Sample:**
```text
✓ Detected: car | Confidence: 91.25%
✓ Detected: person | Confidence: 84.70%

--- OBJECT DETECTION COMPLETE ---
Total Validated Objects (>= 80%): 2
Result Image Saved As: output_detection_result.png
```

---

## 🏁 Milestone Verification Checklist

| Requirement Milestone | Implementation Checkpoint | Status |
| :--- | :--- | :--- |
| **Library Integration** | Implemented `cv2.dnn.readNetFromCaffe()` / `cv2.dnn.readNet()` | PASS |
| **Pre-Processing Integrity** | Applied 4D Blob Construction ($300 \times 300$, scale factor $0.007843$, mean $127.5$) | PASS |
| **Accuracy Benchmarking** | Enforced confidence gate filtering ($C_i \ge 0.80 / 0.50$) | PASS |
| **Visual Confirmation** | Output image exported with bounding boxes & percentage overlay | PASS |

---

## 📝 License
This project is created for educational purposes under the DecodeLabs Architect's Playbook curriculum.