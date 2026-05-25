import numpy as np
import cv2
import torch
from typing import Tuple
from ultralytics import YOLO
from utils.config_manager import SystemConfig


# ==============================================================================
# PERCEPTION TRACKING ENGINE (PRODUCTION OPTIMIZED)
# ==============================================================================

class PlantTracker:
    """
    Handles loading the YOLO model and running active multi-object tracking (MOT).
    Optimized for high-throughput and robotic pipeline integration.
    """

    def __init__(self, config: SystemConfig):
        self.model_path = config.model.path
        self.imgsz = config.model.imgsz
        self.conf_threshold = config.model.conf_threshold
        self.tracker_type = config.model.tracker_type

        # Explicit Device Handling (GPU if available, fallback to CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"⚙️ Loading YOLO Tracker on device: {self.device}...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"✅ Tracker initialized using configuration: {self.tracker_type}")

    def track_frame(
            self,
            frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs segmentation and tracking on a single frame.

        Returns:
            ids (np.ndarray): [N] tracking IDs (int)
            boxes (np.ndarray): [N, 4] bounding boxes (float32)
            masks (np.ndarray): [N, H, W] binary segmentation masks (bool)
            confs (np.ndarray): [N] confidence scores (float32)
            classes (np.ndarray): [N] class IDs (int)
        """
        h, w, _ = frame.shape

        # Run YOLO tracking with persistence enabled (remembers past frames)
        results = self.model.track(
            frame,
            persist=True,  # Critical: keeps tracking state alive
            tracker=self.tracker_type,  # E.g., "bytetrack.yaml"
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        # Production Safety: Return empty structured arrays instead of None.
        # This prevents downstream code from breaking with AttributeError.
        if results.boxes is None or results.boxes.id is None or results.masks is None:
            return (
                np.empty((0,), dtype=int),  # ids
                np.empty((0, 4), dtype=np.float32),  # boxes
                np.empty((0, h, w), dtype=bool),  # masks
                np.empty((0,), dtype=np.float32),  # confs
                np.empty((0,), dtype=int)  # classes
            )

        # 1. Extract Tracking IDs
        ids = results.boxes.id.cpu().numpy().astype(int)

        # 2. Extract Bounding Boxes (xyxy format)
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)

        # 3. Extract Confidences
        confs = results.boxes.conf.cpu().numpy().astype(np.float32)

        # 4. Extract Class IDs
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # 5. Optimized Mask Processing: INTER_NEAREST on uint8 to prevent scaling artifacts
        raw_masks = results.masks.data.cpu().numpy()

        # Vectorized list comprehension for high-speed resizing
        resized_masks = [
            cv2.resize((mask * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 127
            for mask in raw_masks
        ]

        masks_array = np.array(resized_masks, dtype=bool)

        return ids, boxes, masks_array, confs, classes