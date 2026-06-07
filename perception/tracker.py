# perception/tracker.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.config_manager import SystemConfig
import os
import logging # Import the logging module

# Get a logger instance for this module
logger = logging.getLogger(__name__)
# Configure logging if not already configured (e.g., by main.py)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PlantTracker:
    """Perception module responsible for loading the YOLO segmentation model
    and running frame-by-frame Multi-Object Tracking (MOT).

    This class encapsulates model loading, device placement, inference,
    spatial boundary filtering, and temporal box smoothing (EMA).

    Supported Backends (selected automatically by file extension):
        - .pt:     PyTorch (standard PyTorch API, runs on CPU or GPU)
        - .onnx:   ONNX Runtime (high-performance CPU or CUDA Execution Provider)
        - .engine: TensorRT (optimized GPU-only inference, device-locked at build)

    Architectural pipeline flow per frame:
        1. Run YOLO object detection and tracking.
        2. Filter out edge-case detections touching image boundaries.
        3. Smooth bounding boxes temporally using an Exponential Moving Average (EMA).
        4. Resize segmentation masks dynamically (deferred resizing to save compute).
        5. Age inactive tracking IDs to gracefully clean up tracking memory.

    Attributes:
        model (YOLO): Loaded Ultralytics model instance.
        imgsz (int): Input image resolution (squared width/height) for the model.
        conf_threshold (float): Minimum confidence score to accept detections.
        tracker_type (str): Path or identifier of the tracker configuration (e.g., 'bytetrack').
        device (int|str): Runtime hardware target (0 for CUDA GPU, 'cpu' for CPU).
        use_half (bool): Flag indicating if FP16 precision is used for preprocessing.
        track_missing_frames (dict): Maps track ID to inactive frame count (for track aging).
        smoothed_boxes (dict): Maps track ID to its smoothed bounding box coordinate array.
        max_missing_frames (int): Grace period (in frames) before a lost track is purged.
        edge_margin (int): Boundary margin in pixels; plants within this zone are discarded.
        ema_alpha (float): Smoothing factor for Bounding Box EMA. Range: (0.0, 1.0].
    """

    def __init__(self, config: SystemConfig):
        """Initializes the PlantTracker using variables defined in SystemConfig.

        Args:
            config (SystemConfig): Configuration dataclass containing all system hyperparameters.
        """
        # ----------------------------------------------------------------------
        # Configuration Extraction
        # ----------------------------------------------------------------------
        base_model_path = config.model.path
        self.imgsz = config.model.imgsz
        self.conf_threshold = config.model.conf_threshold
        self.tracker_type = config.model.tracker_type

        # ----------------------------------------------------------------------
        # Hardware Target Detection
        # ----------------------------------------------------------------------
        logger.info(f"TRACKER CUDA: {torch.cuda.is_available()}")
        logger.info(f"TRACKER GPU COUNT: {torch.cuda.device_count()}")

        # FP16 acceleration is restricted to CUDA devices (CPUs do not natively support half)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()

        logger.info(f"TRACKER DEVICE: {self.device}")

        # ----------------------------------------------------------------------
        # Model Loading & Device Allocation (Single Source of Truth)
        # ----------------------------------------------------------------------
        logger.info(f"Loading model: {base_model_path}")
        ext = os.path.splitext(base_model_path)[1].lower()

        if ext == ".pt":
            # Native PyTorch weights require explicit device allocation
            self.model = YOLO(base_model_path, task="segment")
            self.model.to(self.device)
        elif ext in [".onnx", ".engine"]:
            # Compiled formats manage device execution contexts internally.
            # Do NOT call .to() on .onnx or .engine to prevent runtime backend conflicts.
            self.model = YOLO(base_model_path, task="segment")
        else:
            logger.error(f"Unsupported model format: {ext}")
            raise ValueError(f"Unsupported model format: {ext}")

        logger.info(f"Tracker initialized using: {self.tracker_type}")

        # ----------------------------------------------------------------------
        # Temporal Tracking Memory Initialization
        # ----------------------------------------------------------------------
        self.track_missing_frames = {}
        self.smoothed_boxes = {}

        self.max_missing_frames = config.model.max_missing_frames
        self.edge_margin = config.model.edge_margin
        self.ema_alpha = config.model.ema_alpha

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def track_frame(self, frame: np.ndarray):
        """Processes a single video frame: tracks entities, filters edge noise,
        smoothes boxes, and dynamically resizes segmentation masks.

        Args:
            frame (np.ndarray): Input image frame of shape [H, W, 3] with dtype uint8.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - ids (np.ndarray): Unique tracking IDs of shape [N], dtype int32.
                - boxes (np.ndarray): Smoothed bounding box coordinates [x1, y1, x2, y2]
                  of shape [N, 4], dtype float32.
                - masks (np.ndarray): Binary segmentation masks of shape [N, H, W], dtype bool.
                - confs (np.ndarray): Model confidence scores of shape [N], dtype float32.
                - classes (np.ndarray): Predicted class IDs of shape [N], dtype int32.
        """
        h, w = frame.shape[:2]

        # STEP 1: Execute YOLO Detection and Tracking
        # half=True leverages FP16 on GPU to accelerate tensor preprocessing.
        # Explicit device mapping prevents Ultralytics from running auto-detection threads.
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
            half=self.use_half,
            device=self.device,
        )[0]

        # ----------------------------------------------------------------------
        # Empty Frame / Tracking Loss Safety Guard
        # ----------------------------------------------------------------------
        # If no objects are tracked, return pre-allocated empty arrays of correct
        # shapes and dtypes to prevent downstream downstream crashes or None checks.
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
                np.empty((0,), dtype=np.int32),
            )

        # ----------------------------------------------------------------------
        # Extract GPU Tensors & Transfer to Host Memory
        # ----------------------------------------------------------------------
        ids = results.boxes.id.cpu().numpy().astype(np.int32)
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = results.boxes.conf.cpu().numpy().astype(np.float32)
        classes = results.boxes.cls.cpu().numpy().astype(np.int32)

        # Get raw low-resolution masks from the model (deferred resizing)
        raw_masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        # ----------------------------------------------------------------------
        # Filtering, Box Smoothing & Lazy Resizing Pipeline
        # ----------------------------------------------------------------------
        filtered_ids = []
        filtered_boxes = []
        filtered_masks = []
        filtered_confs = []
        filtered_classes = []

        for i in range(len(ids)):
            plant_id = int(ids[i])
            x1, y1, x2, y2 = boxes[i]

            # Spatially filter out plants touching the top/bottom borders.
            # Plants at the edge yield cut-off bounding boxes and noisy masks.
            if self._is_at_edge(y1, y2, h, self.edge_margin):
                continue

            # Apply Exponential Moving Average (EMA) to suppress temporal jitter
            smooth_box = self._apply_ema(plant_id, boxes[i])
            self.smoothed_boxes[plant_id] = smooth_box
            self.track_missing_frames[plant_id] = 0  # Reset missing counter on hit

            filtered_ids.append(plant_id)
            filtered_boxes.append(smooth_box)
            filtered_confs.append(confs[i])
            filtered_classes.append(classes[i])

            # PERFORMANCE OPTIMIZATION: "Lazy Mask Resizing"
            # cv2.resize is a highly intensive CPU/GPU operation.
            # We defer resizing and ONLY process masks that survived the boundary filter.
            if raw_masks is not None:
                resized = cv2.resize(
                    (raw_masks[i] > 0.5).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                filtered_masks.append(resized)

        # ----------------------------------------------------------------------
        # Track Aging & Garbage Collection
        # ----------------------------------------------------------------------
        # Increment missing count for active trackers not visible in the current frame.
        # Purge their memory from trackers once the max missing frame limit is crossed.
        active_ids = set(filtered_ids)

        for track_id in list(self.track_missing_frames.keys()):
            if track_id not in active_ids:
                self.track_missing_frames[track_id] += 1
                if self.track_missing_frames[track_id] > self.max_missing_frames:
                    del self.track_missing_frames[track_id]
                    self.smoothed_boxes.pop(track_id, None)

        # ----------------------------------------------------------------------
        # Format and Return Final Data Arrays
        # ----------------------------------------------------------------------
        return (
            np.array(filtered_ids, dtype=np.int32),
            np.array(filtered_boxes, dtype=np.float32),
            np.array(filtered_masks, dtype=bool)
                if filtered_masks
                else np.empty((0, h, w), dtype=bool),
            np.array(filtered_confs, dtype=np.float32),
            np.array(filtered_classes, dtype=np.int32),
        )

    # ==========================================================================
    # HELPER METHODS (Helper/Internal Utilities)
    # ==========================================================================

    def _apply_ema(self, plant_id: int, current_box: np.ndarray) -> np.ndarray:
        """Applies an Exponential Moving Average (EMA) filter to the box coordinates.

        Formula:
            Box_smoothed = alpha * Box_current + (1 - alpha) * Box_previous

        Args:
            plant_id (int): Unique identifier of the tracked object.
            current_box (np.ndarray): Coordinate array [x1, y1, x2, y2] of shape [4].

        Returns:
            np.ndarray: Smoothed coordinate array of shape [4], dtype float32.
        """
        if plant_id in self.smoothed_boxes:
            prev_box = self.smoothed_boxes[plant_id]
            return self.ema_alpha * current_box + (1.0 - self.ema_alpha) * prev_box
        return current_box

    def _is_at_edge(self, y1: float, y2: float, h: int, edge_margin: int) -> bool:
        """Checks if a bounding box is located within the defined border margin.

        This check acts as a noise-gate to filter out partial, truncated objects
        near the vertical borders of the video.

        Args:
            y1 (float): Upper vertical coordinate of the box.
            y2 (float): Lower vertical coordinate of the box.
            h (int): Height of the current image frame in pixels.
            edge_margin (int): Boundary margin width in pixels.

        Returns:
            bool: True if the box coordinates lie within the margin zone.
        """
        return y1 < edge_margin or y2 > h - edge_margin