# Uncertainty-Aware Crop Perception for Agricultural Robotics

## Real-Time Crop and Weed Perception for Agricultural Robotics

This project presents a real-time computer vision pipeline for agricultural robotics that combines instance segmentation, multi-object tracking, temporal state estimation, and rule-based decision making.

The system is designed to distinguish crops from weeds, maintain persistent object identities over time, and support autonomous robotic interventions through uncertainty-aware decision logic.

The project focuses on practical deployment aspects including TensorRT acceleration, performance profiling, automated testing, and modular software architecture.

---

## Demo

The pipeline generates an annotated output video showing:

* Instance segmentation results
* Multi-object tracking IDs
* Decision states
* System telemetry
* Optional uncertainty visualization

![Sample Detection](output.gif)

---

# Highlights

* YOLO26s-seg fine-tuned on PhenoBench
* Real-time crop and weed instance segmentation
* ByteTrack multi-object tracking
* Temporal confidence smoothing
* ROI-based robotic decision engine
* TensorRT FP16 deployment
* 21–22 FPS on NVIDIA Tesla T4
* Detailed pipeline profiling
* 32 automated unit tests
* Modular and configurable architecture

---

# Key Features

## Advanced Perception

The perception layer uses YOLO26s-seg trained on the PhenoBench dataset to perform instance-level crop and weed segmentation.

Features:

* Crop / Weed classification
* Instance segmentation masks
* Confidence estimation
* Real-time inference

---

## Multi-Object Tracking

ByteTrack is used to maintain stable plant identities across video frames.

Features:

* Persistent tracking IDs
* Robust handling of short occlusions
* Temporal consistency for downstream decision logic

---

## Temporal State Memory

Each tracked plant is represented by a digital twin stored in memory.

Features:

* Confidence history
* Sliding confidence window
* Exponential confidence smoothing
* Stability validation
* Grace period for temporary tracking loss

---

## Decision Engine

A rule-based decision layer evaluates every stable plant and assigns one of four states:

| State        | Description                    |
| ------------ | ------------------------------ |
| ALLOW_ACTION | Robot may perform intervention |
| DENY_ACTION  | Intervention blocked           |
| MONITOR      | Continue observation           |
| IGNORE       | Ignore object                  |

Decision logic combines:

* Plant class
* Spatial location
* Confidence score
* Stability criteria
* ROI geofencing

---

## Hardware Acceleration

The system supports multiple deployment backends:

* PyTorch (.pt)
* ONNX Runtime (.onnx)
* TensorRT (.engine)

Automatic model routing enables seamless switching between deployment formats.

---

# Architecture

```text
Video Frame
    │
    ▼
┌─────────────────────────────────────┐
│  Perception Layer                   │
│  YOLO26s-seg + ByteTrack            │
│  .engine → .onnx → .pt routing      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  State Memory                       │
│  TrackedPlant Digital Twin          │
│  Confidence Smoothing               │
│  Grace Period                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Decision Engine                    │
│  ROI Geofencing                     │
│  Confidence Validation              │
│  4-State Logic                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Visualization & Telemetry          │
│  CSV / JSON Metrics                 │
│  Video Output                       │
└─────────────────────────────────────┘
```

---

# Performance Benchmarks

## Benchmark Environment

* GPU: NVIDIA Tesla T4
* Resolution: 960 × 960
* Dataset: PhenoBench
* Precision: FP32 / FP16
* Backend: PyTorch, ONNX Runtime, TensorRT

---

## Best Deployment Configuration

| Model       | Backend  | Precision | FPS  | Latency |
| ----------- | -------- | --------- | ---- | ------- |
| YOLO26s-seg | TensorRT | FP16      | 21.9 | 42.7 ms |

---

## Accuracy Comparison

| Metric       | PT_11S_FP32 | TRT_11S_FP16 | PT_26S_FP32 | TRT_26S_FP16 |
| ------------ | ----------- | ------------ | ----------- | ------------ |
| Box mAP50    | 0.8714      | 0.8642       | 0.8750      | 0.8759       |
| Box mAP50-95 | 0.6734      | 0.6628       | 0.6749      | 0.6743       |
| Seg mAP50    | 0.8454      | 0.8473       | 0.8434      | 0.8510       |
| Seg mAP50-95 | 0.5619      | 0.5547       | 0.5566      | 0.5592       |

### Conclusion

YOLO26s-seg achieved the best overall deployment performance, combining the highest segmentation accuracy with near real-time throughput when deployed using TensorRT FP16.

Complete benchmark tables are available in:

```text
docs/benchmarks.md
```

---

# Profiling Results

Pipeline profiling was performed to identify bottlenecks and optimization opportunities.

Example TensorRT FP16 run:

| Component       | Latency |
| --------------- | ------- |
| Inference       | ~41 ms  |
| Tracking        | <1 ms   |
| Decision Engine | <1 ms   |
| Visualization   | ~70 ms  |
| Total Pipeline  | ~127 ms |

Profiling showed that visualization consumed more runtime than inference itself.

Disabling visualization increased throughput from approximately:

```text
7.9 FPS → 22.5 FPS
```

Detailed profiling results are documented in:

```text
docs/profiling.md
```

---

# Software Quality

The project includes a comprehensive automated testing suite.

## Unit Tests

Total:

```text
32 automated unit tests
```

Coverage:

### tests/test_state.py

* Confidence smoothing
* Sliding window behavior
* Stability validation
* Grace period handling

### tests/test_policy.py

* ROI geofencing
* Decision thresholds
* State transitions
* Multi-object scenarios

### tests/test_tracker.py

* Output validation
* Tracking consistency
* Edge filtering
* Helper utilities

Run tests:

```bash
pytest
```

---

# Dataset

The perception models were trained using the PhenoBench dataset.

Features:

* High-resolution drone imagery
* Crop and weed annotations
* Instance segmentation masks
* Real agricultural field conditions

Classes:

```text
0 → Crop
1 → Weed
```

---

# Project Structure

```text
project/
│
├── main.py
├── main_profile.py
├── configs/
│
├── perception/
│   └── tracker.py
│
├── decision/
│   ├── state.py
│   └── policy.py
│
├── utils/
│   ├── config_manager.py
│   ├── viz.py
│   ├── logging.py
│   └── gpu_helper.py
│
├── tests/
│
├── docs/
│   ├── benchmarks.md
│   └── profiling.md
│
├── notebooks/
│
└── simulation/
```

---

# Configuration

All pipeline parameters can be configured via:

```text
configs/default.yaml
```

Examples:

* Model selection
* Confidence thresholds
* Tracking parameters
* ROI configuration
* Input/output paths
* Visualization settings

---

# Running the Pipeline

Run inference:

```bash
python main.py
```

Run profiling:

```bash
python main_profile.py
```

Run tests:

```bash
pytest
```

---

# Future Work

* Jetson Orin deployment
* TensorRT INT8 quantization
* ROS2 integration
* Multi-camera support
* Field robotics integration

---

# License

MIT License
