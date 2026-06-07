# decision/state.py
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Deque, Dict
from utils.config_manager import MemoryConfig
import math # Import the math module for log2


class InterventionState(Enum):
    """
    Defines the possible intervention states for a tracked plant.
    """
    IGNORE       = 0
    MONITOR      = 1
    ALLOW_ACTION = 2
    DENY_ACTION  = 3


@dataclass
class TrackedPlant:
    """
    Represents a single tracked plant with its state and history.
    """
    plant_id: int
    class_id: int

    conf_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    seen_count:   int   = 0
    missing_count: int  = 0
    is_stable:    bool  = False
    smoothed_conf: float = 0.0
    entropy: float = 0.0 # Add entropy field
    is_in_action_zone: bool = False # New field to indicate if plant is in action zone
    decision_locked: bool = False   # New field to lock decision-making values

    bbox: np.ndarray = None
    mask: np.ndarray = None

    def update_history(self, conf: float, bbox: np.ndarray, mask: np.ndarray, config: MemoryConfig):
        """
        Updates the plant's history with new detection information.
        If decision_locked is True, only bbox, mask, seen_count, and missing_count are updated.
        smoothed_conf and entropy are not updated to preserve the locked state.

        Args:
            conf (float): The confidence score of the current detection.
            bbox (np.ndarray): The bounding box of the current detection.
            mask (np.ndarray): The segmentation mask of the current detection.
            config (MemoryConfig): The memory configuration for tracking.
        """
        self.bbox = bbox
        self.mask = mask
        self.seen_count  += 1
        self.missing_count = 0

        if self.seen_count >= config.min_stable_frames:
            self.is_stable = True

        # Only update smoothed_conf and entropy if the decision is not locked
        if not self.decision_locked:
            if self.conf_history.maxlen != config.window_size:
                self.conf_history = deque(self.conf_history, maxlen=config.window_size)

            self.conf_history.append(conf)

            self.smoothed_conf = sum(self.conf_history) / len(self.conf_history)

            # Calculate entropy based on smoothed_conf
            p = max(1e-9, min(1.0 - 1e-9, self.smoothed_conf))
            self.entropy = (
                -p * math.log2(p)
                - (1.0 - p) * math.log2(1.0 - p)
            )


class StateManager:
    """
    Manages the state of all active tracked plants, including their history and lifecycle.
    """

    def __init__(self, config: MemoryConfig, max_missing_frames: int = 5):
        """
        Initializes the StateManager.

        Args:
            config (MemoryConfig): The memory configuration for tracking.
            max_missing_frames (int): The maximum number of frames a plant can be missing
                                      before it's removed from active tracking.
        """
        self.config = config
        self.max_missing_frames = max_missing_frames
        self.active_plants: Dict[int, TrackedPlant] = {}

    def update_state(self, ids, classes, confs, boxes, masks):
        """
        Updates the state of all tracked plants based on new frame detections.

        Args:
            ids: List of tracking IDs from the current frame.
            classes: List of class IDs from the current frame.
            confs: List of confidence scores from the current frame.
            boxes: List of bounding boxes from the current frame.
            masks: List of segmentation masks from the current frame.
        """
        current_frame_ids = self._update_active_tracks(ids, classes, confs, boxes, masks)
        self._cleanup_lost_tracks(current_frame_ids)

    def _update_active_tracks(self, ids, classes, confs, boxes, masks) -> set:
        """
        Updates existing active tracks and adds new ones.

        Args:
            ids: List of tracking IDs from the current frame.
            classes: List of class IDs from the current frame.
            confs: List of confidence scores from the current frame.
            boxes: List of bounding boxes from the current frame.
            masks: List of segmentation masks from the current frame.

        Returns:
            set: A set of IDs present in the current frame.
        """
        current_frame_ids = set()
        if len(ids) == 0:
            return current_frame_ids

        for i, plant_id in enumerate(ids):
            current_frame_ids.add(plant_id)

            class_id = int(classes[i])

            if plant_id not in self.active_plants:
                self.active_plants[plant_id] = TrackedPlant(
                    plant_id=plant_id,
                    class_id=class_id
                )

            self.active_plants[plant_id].update_history(
                conf=confs[i],
                bbox=boxes[i],
                mask=masks[i] if i < len(masks) else None,
                config=self.config
            )

        return current_frame_ids

    def _cleanup_lost_tracks(self, current_frame_ids: set):
        """
        Removes tracks that have been missing for too many consecutive frames.

        Args:
            current_frame_ids (set): A set of IDs present in the current frame.
        """
        lost_ids = set(self.active_plants.keys()) - current_frame_ids

        for lost_id in list(lost_ids):
            plant = self.active_plants[lost_id]
            plant.missing_count += 1

            if plant.missing_count > self.max_missing_frames:
                del self.active_plants[lost_id]