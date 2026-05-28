# decision/state.py
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import List, Dict
from utils.config_manager import MemoryConfig


# ==============================================================================
# 1. SYSTEM INTERVENTION STATES (ENUM)
# ==============================================================================

class InterventionState(Enum):
    IGNORE = 0  # Non-target instances
    MONITOR = 1  # Target detected but outside the active Region of Interest (ROI)
    ALLOW_ACTION = 2  # Target in ROI with high smoothed confidence (Ready)
    DENY_ACTION = 3  # Target in ROI but smoothed confidence is too low (Safety Abort)


# ==============================================================================
# 2. INDIVIDUAL PLANT MEMORY STATE (DATACLASS)
# ==============================================================================

@dataclass
class TrackedPlant:
    """
    Represents a single tracked plant instance over time (Digital Twin).
    """
    plant_id: int
    class_id: int

    # Tracking lifecycle and smoothing
    conf_history: List[float] = field(default_factory=list)
    seen_count: int = 0
    missing_count: int = 0  # Grace period counter
    is_stable: bool = False
    smoothed_conf: float = 0.0  # Temporally smoothed confidence

    # Spatial data
    bbox: np.ndarray = None  # [x1, y1, x2, y2]
    mask: np.ndarray = None  # Binary mask

    def update_history(self, conf: float, bbox: np.ndarray, mask: np.ndarray, config: MemoryConfig):
        """
        Updates tracking metrics, spatial data, and temporal confidence smoothing.
        """
        self.bbox = bbox
        self.mask = mask
        self.seen_count += 1
        self.missing_count = 0  # Reset missing counter when detected

        # Anti-Flicker check
        if self.seen_count >= config.min_stable_frames:
            self.is_stable = True

        # Update confidence history (Sliding Window)
        self.conf_history.append(conf)
        if len(self.conf_history) > config.window_size:
            self.entropy_history = self.conf_history.pop(0)

        # Smooth confidence over the temporal window
        self.smoothed_conf = float(np.mean(self.conf_history))


# ==============================================================================
# 3. STATE MANAGER
# ==============================================================================

class StateManager:
    """
    Manages the lifecycle of active tracked plants.
    Implements a trace grace period to prevent track loss during short occlusions.
    """

    def __init__(self, config: MemoryConfig, max_missing_frames: int = 5):
        self.config = config
        self.max_missing_frames = max_missing_frames
        self.active_plants: Dict[int, TrackedPlant] = {}

    def update_state(
            self,
            ids: np.ndarray,
            classes: np.ndarray,
            confs: np.ndarray,
            boxes: np.ndarray,
            masks: np.ndarray
    ):
        """
        Orchestrates the state update pipeline (SRP compliant).
        """
        current_frame_ids = self._update_active_tracks(ids, classes, confs, boxes, masks)
        self._cleanup_lost_tracks(current_frame_ids)

    def _update_active_tracks(self, ids, classes, confs, boxes, masks) -> set:
        """
        Helper: Registers new tracks and updates existing ones with spatial data.
        """
        current_frame_ids = set()
        if len(ids) == 0:
            return current_frame_ids

        for i, plant_id in enumerate(ids):
            current_frame_ids.add(plant_id)

            # Extract current frame features
            conf = confs[i]
            bbox = boxes[i]
            mask = masks[i]
            class_id = int(classes[i])

            # Register new plant if not in memory
            if plant_id not in self.active_plants:
                self.active_plants[plant_id] = TrackedPlant(
                    plant_id=plant_id,
                    class_id=class_id
                )

            # Update temporal history and smooth features
            self.active_plants[plant_id].update_history(
                conf=conf,
                bbox=bbox,
                mask=mask,
                config=self.config
            )

        return current_frame_ids

    def _cleanup_lost_tracks(self, current_frame_ids: set):
        """
        Helper: Implements a grace period before deleting lost tracks from RAM.
        """
        lost_ids = set(self.active_plants.keys()) - current_frame_ids

        for lost_id in list(lost_ids):
            plant = self.active_plants[lost_id]
            plant.missing_count += 1

            # Delete only if the plant has been missing for too many frames
            if plant.missing_count > self.max_missing_frames:
                del self.active_plants[lost_id]