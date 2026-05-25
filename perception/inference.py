import numpy as np
import cv2
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

        # 5. Device Handling: Explicitly assign model to GPU/CPU
        self.device = "cuda:0" if config.model.tracker_type else "cpu"  # or from config

        print(f"⚙️ Loading YOLO Tracker on device: {self.device}...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"✅ Tracker initialized using: {self.tracker_type}")

    def track_frame(
            self,
            frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs segmentation and tracking on a single frame.

        6. Explicit Typing: Returns structured numpy arrays instead of raw python objects.
        Returns:
            ids (np.ndarray): [N] tracking IDs (int)
            boxes (np.ndarray): [N, 4] bounding boxes (float32)
            masks (np.ndarray): [N, H, W] binary segmentation masks (bool)
            confs (np.ndarray): [N] confidence scores (float32)
            classes (np.ndarray): [N] class IDs (int)
        """
        h, w, _ = frame.shape

        # Run YOLO tracking with persistence enabled
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        # 4. Safe Fallbacks : Return empty structured arrays instead of None
        # This prevents downstream code from breaking with AttributeError or NoneType checks.
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

        # 2. Extract Bounding Boxes
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)

        # 3. Extract Confidences
        confs = results.boxes.conf.cpu().numpy().astype(np.float32)

        # 4. Extract Class IDs
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # 5. Optimized Mask Processing: INTER_NEAREST on uint8 to prevent scaling artifacts
        raw_masks = results.masks.data.cpu().numpy()

        # Fast list comprehension resizing
        resized_masks = [
            cv2.resize((mask * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 127
            for mask in raw_masks
        ]

        masks_array = np.array(resized_masks, dtype=bool)

        return ids, boxes, masks_array, confs, classes