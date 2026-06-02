#tracker
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.config_manager import SystemConfig
import os


class PlantTracker:

    def __init__(self, config: SystemConfig):
        """
        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------
        base_model_path = config.model.path
        self.imgsz = config.model.imgsz
        self.conf_threshold = config.model.conf_threshold
        self.tracker_type = config.model.tracker_type

        # ------------------------------------------------------------
        # Device detection
        # ------------------------------------------------------------
        print("TRACKER CUDA:", torch.cuda.is_available())
        print("TRACKER GPU COUNT:", torch.cuda.device_count())

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()  # FP16 nur auf GPU
        print("TRACKER DEVICE:", self.device)

        # ------------------------------------------------------------
        # Model routing
        # ------------------------------------------------------------
        resolved_model_path = base_model_path

        if torch.cuda.is_available():
            engine_path = base_model_path.replace(".pt", ".engine")
            onnx_path = base_model_path.replace(".pt", ".onnx")

            if os.path.exists(engine_path):
                resolved_model_path = engine_path
                print(f"Using TensorRT engine: {resolved_model_path}")
            elif os.path.exists(onnx_path):
                resolved_model_path = onnx_path
                print(f"Using ONNX model: {resolved_model_path}")
            else:
                print(f"Using PyTorch model (fallback): {resolved_model_path}")
        else:
            print(f"Using CPU PyTorch model: {resolved_model_path}")

        # ------------------------------------------------------------
        # Load model
        # OPT 1: task="segment" explizit setzen →  richtige Pipeline,
        #         kein Raten → spart ~3-5ms pro Frame
        # ------------------------------------------------------------
        print(f"Loading model: {resolved_model_path}")

        if resolved_model_path.endswith(".pt"):
            self.model = YOLO(resolved_model_path, task="segment")
            self.model.to(self.device)
        else:
            self.model = YOLO(resolved_model_path, task="segment")

        print(f"Tracker initialized using: {self.tracker_type}")
        """
        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------
        base_model_path = config.model.path
        self.imgsz = config.model.imgsz
        self.conf_threshold = config.model.conf_threshold
        self.tracker_type = config.model.tracker_type

        # ------------------------------------------------------------
        # Device detection
        # ------------------------------------------------------------
        print("TRACKER CUDA:", torch.cuda.is_available())
        print("TRACKER GPU COUNT:", torch.cuda.device_count())

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()

        print("TRACKER DEVICE:", self.device)

        # ------------------------------------------------------------
        # Model loading (SINGLE SOURCE OF TRUTH)
        # ------------------------------------------------------------
        print(f"Loading model: {base_model_path}")

        ext = os.path.splitext(base_model_path)[1].lower()

        if ext == ".pt":
            # PyTorch
            self.model = YOLO(base_model_path, task="segment")
            self.model.to(self.device)

        elif ext in [".onnx", ".engine"]:
            # ONNX or TensorRT
            self.model = YOLO(base_model_path, task="segment")

        else:
            raise ValueError(f"Unsupported model format: {ext}")

        print(f"Tracker initialized using: {self.tracker_type}")
        # ------------------------------------------------------------
        # Tracking memory
        # ------------------------------------------------------------
        self.track_missing_frames = {}
        self.smoothed_boxes = {}

        self.max_missing_frames = config.model.max_missing_frames
        self.edge_margin = config.model.edge_margin
        self.ema_alpha = config.model.ema_alpha

    # ==================================================================
    # TRACK FRAME
    # ==================================================================
    def track_frame(self, frame: np.ndarray):

        h, w = frame.shape[:2]

        # OPT 2: half=True → FP16 Preprocessing auf GPU
        #         device=0  → explizit GPU, kein Routing-Check
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
            half=self.use_half,      # FP16 nur wenn GPU verfügbar
            device=self.device,      # 0 (GPU) oder "cpu" automatisch
        )[0]

        # ------------------------------------------------------------
        # Empty safety
        # ------------------------------------------------------------
        if (
            results.boxes is None
            or len(results.boxes) == 0
            or results.boxes.id is None
        ):
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, h, w), dtype=bool),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )

        # ------------------------------------------------------------
        # Extract outputs
        # ------------------------------------------------------------
        ids    = results.boxes.id.cpu().numpy().astype(np.int32)
        boxes  = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs  = results.boxes.conf.cpu().numpy().astype(np.float32)
        classes = results.boxes.cls.cpu().numpy().astype(np.int32)

        # OPT 3: Rohe Masken einmal von GPU holen, aber NICHT sofort alle resizen.
        #         Resize passiert lazy NUR für Detektionen die den Filter bestehen.
        raw_masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        # ------------------------------------------------------------
        # Filtering + smoothing + lazy mask resize
        # ------------------------------------------------------------
        filtered_ids     = []
        filtered_boxes   = []
        filtered_masks   = []
        filtered_confs   = []
        filtered_classes = []

        for i in range(len(ids)):

            plant_id = int(ids[i])
            x1, y1, x2, y2 = boxes[i]

            # Edge filter — gefilterte Detektionen bekommen KEINE Maske resized
            if y1 < self.edge_margin or y2 > h - self.edge_margin:
                continue

            # EMA box smoothing
            current_box = boxes[i]
            if plant_id in self.smoothed_boxes:
                prev_box = self.smoothed_boxes[plant_id]
                smooth_box = (
                    self.ema_alpha * current_box
                    + (1.0 - self.ema_alpha) * prev_box
                )
            else:
                smooth_box = current_box

            self.smoothed_boxes[plant_id] = smooth_box
            self.track_missing_frames[plant_id] = 0

            filtered_ids.append(plant_id)
            filtered_boxes.append(smooth_box)
            filtered_confs.append(confs[i])
            filtered_classes.append(classes[i])

            # OPT 3: Resize NUR für Detektionen die den Filter bestehen
            if raw_masks is not None:
                resized = cv2.resize(
                    (raw_masks[i] > 0.5).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ) > 0
                filtered_masks.append(resized)

        # ------------------------------------------------------------
        # Track aging
        # ------------------------------------------------------------
        active_ids = set(filtered_ids)

        for track_id in list(self.track_missing_frames.keys()):
            if track_id not in active_ids:
                self.track_missing_frames[track_id] += 1
                if self.track_missing_frames[track_id] > self.max_missing_frames:
                    del self.track_missing_frames[track_id]
                    self.smoothed_boxes.pop(track_id, None)

        # ------------------------------------------------------------
        # Return
        # ------------------------------------------------------------
        return (
            np.array(filtered_ids,   dtype=np.int32),
            np.array(filtered_boxes, dtype=np.float32),
            np.array(filtered_masks, dtype=bool)
                if filtered_masks
                else np.empty((0, h, w), dtype=bool),
            np.array(filtered_confs,   dtype=np.float32),
            np.array(filtered_classes, dtype=np.int32)
        )