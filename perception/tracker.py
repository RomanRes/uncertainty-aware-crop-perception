import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.config_manager import SystemConfig


# ==============================================================================
# PLANT TRACKER
# ==============================================================================
# Responsibilities:
# - YOLO segmentation inference
# - Multi-object tracking (ByteTrack)
# - Stable track lifecycle handling
# - Edge filtering
# - Bounding box smoothing
# - Structured pipeline output
# ==============================================================================


class PlantTracker:
    """
    Production-oriented crop/weed tracking system.

    Features:
    - YOLO segmentation + tracking in one pass
    - Stable tracking IDs
    - Track memory aging
    - Edge filtering
    - EMA bounding box smoothing
    - Safe outputs for downstream robotics pipeline
    """

    def __init__(self, config: SystemConfig):

        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------
        self.model_path = config.model.path
        self.imgsz = config.model.imgsz
        self.conf_threshold = config.model.conf_threshold
        self.tracker_type = config.model.tracker_type

        # ------------------------------------------------------------
        # Device
        # ------------------------------------------------------------
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # ------------------------------------------------------------
        # YOLO Model
        # ------------------------------------------------------------
        print(f"Loading tracker on device: {self.device}")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        #if self.device.startswith("cuda"):
            #self.model.model.half()

        print(f"Tracker initialized using: {self.tracker_type}")

        # ------------------------------------------------------------
        # Tracking Memory
        # ------------------------------------------------------------

        # Stores missing-frame counters
        self.track_missing_frames = {}

        # Stores smoothed boxes
        self.smoothed_boxes = {}

        # ------------------------------------------------------------
        # Tracking Parameters
        # ------------------------------------------------------------

        # How long tracks survive without detection
        self.max_missing_frames = config.model.max_missing_frames

        # Ignore detections near image border
        self.edge_margin = config.model.edge_margin

        # EMA smoothing factor
        self.ema_alpha = config.model.ema_alpha

    # ==========================================================================
    # TRACK FRAME
    # ==========================================================================

    def track_frame(self, frame: np.ndarray):

        h, w = frame.shape[:2]

        # ------------------------------------------------------------
        # YOLO Tracking
        # ------------------------------------------------------------
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        # ------------------------------------------------------------
        # Empty Detection Safety
        # ------------------------------------------------------------
        if (
            results.boxes is None
            or len(results.boxes) == 0
            or results.boxes.id is None
        ):
            return (
                np.empty((0,), dtype=np.int32),         # ids
                np.empty((0, 4), dtype=np.float32),     # boxes
                np.empty((0, h, w), dtype=bool),        # masks
                np.empty((0,), dtype=np.float32),       # confs
                np.empty((0,), dtype=np.int32)          # classes
            )

        # ------------------------------------------------------------
        # Extract Raw Outputs
        # ------------------------------------------------------------
        ids = results.boxes.id.cpu().numpy().astype(np.int32)

        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)

        confs = results.boxes.conf.cpu().numpy().astype(np.float32)

        classes = results.boxes.cls.cpu().numpy().astype(np.int32)

        # ------------------------------------------------------------
        # Segmentation Masks
        # ------------------------------------------------------------
        if results.masks is not None:

            raw_masks = results.masks.data.cpu().numpy()

            masks = np.array([
                cv2.resize(
                    (mask > 0.5).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ) > 0
                for mask in raw_masks
            ], dtype=bool)

        else:
            masks = np.empty((0, h, w), dtype=bool)

        # ------------------------------------------------------------
        # Filtering + Smoothing
        # ------------------------------------------------------------
        filtered_ids = []
        filtered_boxes = []
        filtered_masks = []
        filtered_confs = []
        filtered_classes = []

        for i in range(len(ids)):

            plant_id = int(ids[i])

            x1, y1, x2, y2 = boxes[i]

            # --------------------------------------------------------
            # Edge Filtering
            # Ignore unstable detections near borders
            # --------------------------------------------------------
            if (
                #x1 < self.edge_margin
                #or
                y1 < self.edge_margin
                #or x2 > w - self.edge_margin
                or y2 > h - self.edge_margin
            ):
                continue

            # --------------------------------------------------------
            # EMA Bounding Box Smoothing
            # --------------------------------------------------------
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

            # --------------------------------------------------------
            # Track Aging Reset
            # --------------------------------------------------------
            self.track_missing_frames[plant_id] = 0

            # --------------------------------------------------------
            # Append Stable Output
            # --------------------------------------------------------
            filtered_ids.append(plant_id)

            filtered_boxes.append(smooth_box)

            filtered_confs.append(confs[i])

            filtered_classes.append(classes[i])

            if len(masks) > 0:
                filtered_masks.append(masks[i])

        # ------------------------------------------------------------
        # Track Aging Update
        # ------------------------------------------------------------
        active_ids = set(filtered_ids)

        for track_id in list(self.track_missing_frames.keys()):

            if track_id not in active_ids:

                self.track_missing_frames[track_id] += 1

                # Remove dead tracks
                if (
                    self.track_missing_frames[track_id]
                    > self.max_missing_frames
                ):
                    del self.track_missing_frames[track_id]

                    if track_id in self.smoothed_boxes:
                        del self.smoothed_boxes[track_id]

        # ------------------------------------------------------------
        # Final Structured Output
        # ------------------------------------------------------------
        return (
            np.array(filtered_ids, dtype=np.int32),

            np.array(filtered_boxes, dtype=np.float32),

            np.array(filtered_masks, dtype=bool)
            if len(filtered_masks) > 0
            else np.empty((0, h, w), dtype=bool),

            np.array(filtered_confs, dtype=np.float32),

            np.array(filtered_classes, dtype=np.int32)
        )