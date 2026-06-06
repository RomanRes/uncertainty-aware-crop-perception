# Uncertainty-Aware Crop Perception for Agricultural Robotics

## Revolutionizing Precision Agriculture with Intelligent Robotics

This project introduces a cutting-edge, real-time perception engine designed for advanced agricultural robotics. By leveraging state-of-the-art computer vision and robust decision-making algorithms, we empower autonomous systems to accurately identify and differentiate between crops and weeds, enabling highly precise and efficient interventions. This technology is crucial for reducing herbicide use, optimizing resource allocation, and fostering sustainable farming practices.

---

## Demo

The pipeline generates an output video with a side-by-side view, showcasing the annotated frame (left) and an uncertainty visualization canvas (right), providing clear insights into the system's real-time operation and decision-making.

![Sample Detection](output.gif)

---

## Key Innovations & Business Value

Our perception engine is built on a foundation of robust, real-time performance and intelligent decision-making, offering significant advantages for modern agriculture:

*   **Enhanced Precision:** Accurate instance segmentation and multi-object tracking ensure that robotic actions are directed precisely at target plants, minimizing damage to crops and maximizing weed removal efficiency.
*   **Operational Efficiency:** Real-time processing capabilities allow for immediate decision-making in dynamic field conditions, leading to faster operations and increased throughput for agricultural robots.
*   **Resource Optimization:** By precisely identifying weeds, the system enables targeted herbicide application, significantly reducing chemical usage and associated environmental impact and costs.
*   **Increased Autonomy & Safety:** The uncertainty-aware decision engine provides a critical layer of intelligence, preventing actions in ambiguous situations and ensuring safer, more reliable robotic operations.
*   **Scalability:** Designed with performance in mind, the system can be deployed on various hardware platforms, from embedded systems to cloud-based solutions, adapting to diverse agricultural needs.

---

## Core Features

*   **Advanced Perception:** Utilizes **YOLO11s-seg** for instance segmentation, fine-tuned on the challenging PhenoBench dataset to distinguish between crops and weeds.
*   **Robust Multi-Object Tracking (MOT):** Employs **ByteTrack** to maintain persistent IDs for plants across video frames, crucial for temporal analysis and decision-making.
*   **Hardware Acceleration:** Achieves exceptional speed with **TensorRT FP16** acceleration, delivering pure inference times as low as ~28ms on a Tesla T4 GPU.
*   **Temporal Confidence Smoothing:** Integrates an Exponential Moving Average (EMA) with a configurable sliding window to smooth detection confidences, enhancing tracking stability and reducing false positives/negatives.
*   **Intelligent Decision Engine:** A sophisticated 4-state logic (`ALLOW_ACTION`, `DENY_ACTION`, `MONITOR`, `IGNORE`) guides robotic interventions based on plant type, location, and confidence levels.
*   **Dynamic ROI Geofencing:** Restricts robotic actions to a defined "action zone" within the camera's view, preventing unintended interventions outside critical areas.
*   **Anti-Flicker Stability Guard:** Ensures tracking stability by requiring plants to be consistently observed over 'N' frames before being considered "stable" for decision-making.
*   **Grace Period for Lost Tracks:** Tracks can survive short-term occlusions without ID loss, improving continuity and robustness in challenging environments.
*   **Flexible Model Deployment:** Features automatic model routing, seamlessly loading `.engine` (TensorRT), `.onnx` (ONNX Runtime), or `.pt` (PyTorch) models based on availability and performance needs.
*   **Comprehensive Telemetry & Logging:** Captures frame-level performance metrics (inference time, FPS, GPU utilization) and decision outcomes, providing detailed CSV and JSON summaries for analysis and optimization.

---

## Architectural Overview

The pipeline is designed for modularity and efficiency, processing video frames through distinct layers:

```
Video Frame
    │
    ▼
┌─────────────────────────────────────┐
│  Perception Layer (perception/)     │
│  YOLO11s-seg + ByteTrack            │
│  Auto-routing: .engine → .onnx → .pt│
└──────────────┬──────────────────────┘
               │  ids, boxes, masks, confs, classes
               ▼
┌─────────────────────────────────────┐
│  State Memory (decision/state.py)   │
│  TrackedPlant Digital Twin          │
│  EMA smoothing + Grace Period       │
└──────────────┬──────────────────────┘
               │  active_plants
               ▼
┌─────────────────────────────────────┐
│  Decision Engine (decision/policy.py)│
│  ROI Geofencing + Confidence Check  │
│  ALLOW / DENY / MONITOR / IGNORE    │
└──────────────┬──────────────────────┘
               │  decisions
               ▼
┌─────────────────────────────────────┐
│  Visualization + Telemetry Logger   │
│  Annotated video + CSV/JSON metrics │
└─────────────────────────────────────┘
```

---

## Performance Benchmarks

Achieving real-time performance is paramount for robotic applications. Our system demonstrates superior speed, particularly with hardware-accelerated backends.

**Measurement Environment:**
*   **GPU:** NVIDIA Tesla T4
*   **Input Resolution:** 960 × 960 pixels
*   **Dataset:** PhenoBench drone footage

| Backend       | Full `predict()` Latency | Pure Inference Latency | Notes                               |
| ------------- | ------------------------ | ---------------------- | ----------------------------------- |
| PyTorch (.pt) | ~90 ms                   | ~55 ms                 | Baseline performance.               |
| ONNX Runtime  | ~54 ms                   | ~50 ms                 | Leveraging CUDAExecutionProvider.   |
| TensorRT FP16 | ~48 ms                   | **~28 ms**             | Highly optimized, requires TRT 11.  |

> **Insight:** TensorRT pure inference is approximately 2× faster than ONNX. The remaining gap in full `predict()` latency is primarily due to pre/post-processing overhead at 960px resolution.

---

## Decision States

| State          | Condition                                  | Action                                  |
| -------------- | ------------------------------------------ | --------------------------------------- |
| `ALLOW_ACTION` | Crop in ROI, confidence ≥ threshold        | Robot may proceed with intervention     |
| `DENY_ACTION`  | Crop in ROI, confidence < threshold        | Intervention blocked (safety protocol)  |
| `MONITOR`      | Crop outside ROI                           | Track only, no action initiated         |
| `IGNORE`       | Any weed detected (Class 1)                | Always ignored for crop-specific actions|

---

## Testing & Reliability

The project emphasizes robust testing to ensure reliability and maintainability. A comprehensive suite of unit tests covers critical components of the perception and decision-making pipeline.

**Unit Test Summary:**
*   **Total Tests:** 32 unit tests across 3 modules.
*   **Independence:** Tests are designed to run without a GPU or a real model, using fully mocked backends for speed and consistency.

**Coverage:**
*   `tests/test_state.py` (13 tests): Validates EMA smoothing, sliding window behavior, and grace period logic for `TrackedPlant` and `StateManager`.
*   `tests/test_policy.py` (10 tests): Verifies ROI geofencing, confidence thresholding, and multi-plant decision states within the `DecisionEngine`.
*   `tests/test_tracker.py` (9 tests): Ensures correct output format, EMA helper functionality, and edge filtering in the `PlantTracker`.

**To run tests:**

```sh
pytest
```

---

## Project Structure

The repository is organized for clarity, modularity, and ease of development:

```
uncertainty-aware-crop-perception/
│
├── main.py                    # Main entry point for the perception pipeline
├── main_profile.py            # Entry point for detailed performance profiling
├── kaggle_run.py              # Script for running on Kaggle environments
├── onnx_export.py             # Utility for exporting PyTorch models to ONNX format
├── pytest.ini                 # Pytest configuration
│
├── configs/
│   └── default.yaml           # Centralized configuration for all pipeline parameters
│
├── perception/
│   └── tracker.py             # Manages YOLO model, ByteTrack, and mask processing
│
├── decision/
│   ├── state.py               # Defines TrackedPlant data model and StateManager for temporal memory
│   └── policy.py              # Implements the DecisionEngine with ROI geofencing and 4-state logic
│
├── utils/
│   ├── config_manager.py      # Handles loading and parsing YAML configurations into typed dataclasses
│   ├── viz.py                 # Responsible for rendering annotated frames and uncertainty visualizations
│   ├── logging.py             # System for logging telemetry data to CSV and JSON
│   └── gpu_helper.py          # Monitors NVIDIA GPU utilization in real-time
│
├── tests/                     # Contains all unit tests for core modules
├── notebooks/                 # Jupyter notebooks for experimentation, training, and profiling
├── docs/                      # Documentation, architectural diagrams, and detailed explanations
└── simulation/                # Environment for simulating agricultural field conditions
```

---

## Dataset

The perception models are trained on the **PhenoBench** dataset, a large-scale, high-resolution dataset specifically designed for plant instance segmentation in agricultural fields.

*   **Classes:** `Crop` (0) and `Weed` (1)
*   **Imagery:** High-resolution top-down drone footage
*   **Annotations:** Instance-level segmentation masks for precise learning

---

## Setup & Usage

### Requirements

Ensure you have Python 3.8+ installed. Then, install the necessary dependencies:

```sh
pip install ultralytics opencv-python torch pynvml pyyaml pytest numpy
```

For TensorRT export (typically on Linux with NVIDIA GPUs and TRT 11):

```sh
pip install ultralytics lap onnxruntime-gpu onnxslim
```

### Configuration

All configurable parameters for the pipeline are managed in `configs/default.yaml`. This includes model paths, confidence thresholds, tracking parameters, decision logic, and I/O settings.

```yaml
model:
  path: "models/phenobench_cropweed_seg_yolo11s_960.engine" # Prioritize .engine, then .onnx, then .pt
  imgsz: 960
  conf_threshold: 0.15          # Low threshold to feed uncertainty logic
  tracker_type: "bytetrack.yaml"
  max_missing_frames: 10        # How long tracks survive without detection (Grace Period)
  edge_margin: 20               # Ignore detections near image border (in pixels)
  ema_alpha: 0.7                # Exponential Moving Average smoothing factor (0.0 to 1.0)

memory:
  window_size: 5                # Frames for temporal uncertainty smoothing
  min_stable_frames: 3          # Frame count before a plant is marked as stable

decision:
  entropy_threshold: 0.3        # Max Shannon Entropy allowed before blocking action
  action_zone_ratio: 0.5        # Lower 50% of the screen is the active "Safe Zone"

io:
  input_video: "vertical_drone_flight.mp4"
  output_video: "vertical_advanced_agricultural_engine.mp4"
  metrics_output_csv: "data/run_metrics.csv"
  metrics_output_json: "data/metrics.json"
  max_frames: -1                # Set to -1 to process the entire video
  side_by_side: true            # TRUE = Double-width with Heatmap | FALSE = Standard Single-width
  show_telemetry: true          # Display real-time telemetry on the output video
```

### Running the Pipeline

To execute the perception pipeline:

```sh
python main.py
```

### Running with Profiling

For detailed performance analysis of each pipeline step:

```sh
python main_profile.py
```

### Exporting to ONNX / TensorRT

The `onnx_export.py` script facilitates converting the PyTorch model to ONNX. For TensorRT, specific steps might be required depending on your environment and TensorRT version (e.g., patching Ultralytics for TRT 11 compatibility as shown in the original README).

```python
# Example for ONNX export
from ultralytics import YOLO
model = YOLO("models/phenobench_cropweed_seg_yolo11s_960.pt")
model.export(format="onnx", opset=13, imgsz=960, dynamic=True)
```

---

## Roadmap

Our commitment to continuous improvement drives the following development roadmap:

*   **v1.0 — Functional Baseline:** (Completed) Implemented YOLO11s-seg, ByteTrack, and TensorRT FP16 integration.
*   **Profiling Enhancements:** Integrate advanced profiling tools (e.g., `py-spy`) for per-function millisecond breakdown to identify further optimization opportunities.
*   **Quantitative Validation:** Conduct thorough validation with mAP@50 / mAP@50-95 metrics on the PhenoBench test set to quantify segmentation accuracy.
*   **Latency Target Achievement:** Aggressively optimize the full pipeline to consistently achieve sub-45ms/frame processing for high-speed applications.
*   **INT8 Quantization:** Explore and implement INT8 quantization for further performance gains and reduced memory footprint, pending TensorRT and Ultralytics compatibility.

---

## License

This project is licensed under the MIT License.
