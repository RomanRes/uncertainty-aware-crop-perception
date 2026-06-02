######################################################## # Uncertainty-Aware Crop Perception

A real-time, uncertainty-aware perception engine for agricultural robotics. Performs instance segmentation and multi-object tracking of crops and weeds with temporal confidence smoothing and ROI-based decision making, trained on the [PhenoBench](https://www.phenobench.org/) dataset.

---

## Demo

> Pipeline output: side-by-side view with annotated frame (left) and uncertainty canvas (right).

![Sample Detection](output.gif)

---

## Architecture

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

## Key Features

- **YOLO11s-seg** instance segmentation fine-tuned on PhenoBench (crop / weed classes)
- **ByteTrack** multi-object tracking with persistent IDs across frames
- **TensorRT FP16** acceleration — ~28ms pure inference on Tesla T4
- **Temporal confidence smoothing** via sliding window EMA (configurable window size)
- **4-state decision engine**: `ALLOW_ACTION`, `DENY_ACTION`, `MONITOR`, `IGNORE`
- **ROI geofencing** — only plants in the active action zone trigger intervention
- **Anti-flicker stability guard** — plants must be seen N frames before rendering
- **Grace period** — tracks survive short occlusions without ID loss
- **Auto model routing** — loads `.engine` → `.onnx` → `.pt` automatically
- **Frame-level telemetry** — CSV + JSON summary per run

---

## Project Structure

```
uncertainty-aware-crop-perception/
│
├── main.py                    # Pipeline entry point
├── main_profile.py            # Entry point with cProfile instrumentation
├── kaggle_run.py              # Kaggle notebook runner
├── onnx_export.py             # TensorRT FP16 export script (TRT 11 compatible)
├── pytest.ini
│
├── configs/
│   └── default.yaml           # All pipeline parameters
│
├── perception/
│   └── tracker.py             # PlantTracker — YOLO + ByteTrack + mask processing
│
├── decision/
│   ├── state.py               # TrackedPlant dataclass + StateManager
│   └── policy.py              # DecisionEngine — ROI geofencing + 4-state logic
│
├── utils/
│   ├── config_manager.py      # Typed dataclass config loader (YAML → SystemConfig)
│   ├── viz.py                 # Annotated frame renderer + uncertainty canvas
│   ├── logging.py             # CSV + JSON telemetry logger
│   └── gpu_helper.py          # NVML GPU utilization monitor
│
├── tests/
│   ├── conftest.py
│   ├── test_state.py          # 13 unit tests — TrackedPlant + StateManager
│   ├── test_policy.py         # 10 unit tests — ROI geofencing + decision states
│   └── test_tracker.py        # 9 unit tests — output format, EMA, edge filter
│
├── notebooks/                 # Training + export notebooks (Kaggle)
├── docs/                      # Architecture diagrams
└── simulation/                # Simulation environment (field_env.py)
```

---

## Performance Benchmarks

Measured on **Tesla T4**, input resolution **960 × 960**, PhenoBench drone footage.


| Backend       | Full`predict()` | Pure Inference | Notes                           |
| ------------- | --------------- | -------------- | ------------------------------- |
| PyTorch`.pt`  | ~90 ms          | ~55 ms         | FP32 baseline                   |
| ONNX Runtime  | ~54 ms          | ~50 ms         | CUDAExecutionProvider           |
| TensorRT FP16 | ~48 ms          | **~28 ms**     | TRT 11, patched for BuilderFlag |

> TensorRT pure inference is ~2× faster than ONNX. The gap in full `predict()` is due to pre/postprocessing overhead at 960px.

---

## Setup

### Requirements

```bash
pip install ultralytics opencv-python torch pynvml pyyaml pytest
```

For TensorRT export (Kaggle / Linux with TRT 11):

```bash
pip install ultralytics lap onnxruntime-gpu onnxslim
```

### Configuration

All parameters are in `configs/default.yaml`:

```yaml
model:
  path: "models/phenobench_cropweed_seg_yolo11s_960.engine"
  imgsz: 960
  conf_threshold: 0.15
  tracker_type: "bytetrack.yaml"
  max_missing_frames: 10
  edge_margin: 20
  ema_alpha: 0.7

memory:
  window_size: 5
  min_stable_frames: 3

decision:
  action_zone_ratio: 0.5

io:
  input_video: "vertical_drone_flight.mp4"
  output_video: "output.mp4"
  max_frames: -1
  side_by_side: true
```

### Run

```bash
python main.py
```

### Export TensorRT Engine (TRT 11)

```python
# Apply BuilderFlag patch first (TRT 11 removed FP16 flag)
filepath = "/usr/local/lib/python3.12/dist-packages/ultralytics/utils/export/engine.py"
with open(filepath, "r") as f:
    content = f.read()
with open(filepath, "w") as f:
    f.write(content.replace(
        "config.set_flag(trt.BuilderFlag.FP16)",
        "pass  # TRT11: FP16 enabled by default"
    ))

# Export
from ultralytics import YOLO
model = YOLO("models/phenobench_cropweed_seg_yolo11s_960.pt")
model.export(format="engine", imgsz=960, half=True, device=0, task="segment")
```

---

## Decision States


| State          | Condition                            | Action                        |
| -------------- | ------------------------------------ | ----------------------------- |
| `ALLOW_ACTION` | Crop in ROI, confidence ≥ threshold | Robot may intervene           |
| `DENY_ACTION`  | Crop in ROI, confidence < threshold  | Intervention blocked (safety) |
| `MONITOR`      | Crop outside ROI                     | Track only, no action         |
| `IGNORE`       | Any weed                             | Always ignored                |

---

## Tests

```bash
pytest
```

32 unit tests across 3 modules — no GPU or real model required (fully mocked).

```
tests/test_state.py    — 13 tests: EMA smoothing, sliding window, grace period
tests/test_policy.py   — 10 tests: ROI geofencing, threshold boundary, multi-plant
tests/test_tracker.py  —  9 tests: output format, EMA helper, edge filter
```

---

## Dataset

Trained on **[PhenoBench](https://www.phenobench.org/)** — a large-scale dataset for plant instance segmentation in agricultural fields.

- 2 classes: `Crop` (0), `Weed` (1)
- High-resolution top-down drone imagery
- Instance-level segmentation masks

---

## Roadmap

- [X]  v1.0 — Functional baseline (YOLO11s-seg + ByteTrack + TensorRT FP16)
- [ ]  Profiling — per-function ms breakdown with py-spy
- [ ]  Validation — mAP@50 / mAP@50-95 on PhenoBench test set
- [ ]  Latency target — reduce full pipeline below 45ms/frame
- [ ]  INT8 quantization (pending TRT 11 + Ultralytics compatibility)

---

## License

MIT
