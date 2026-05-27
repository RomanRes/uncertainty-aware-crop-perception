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
    """
    Standardized states for the robotic decision and intervention pipeline.
    """
    IGNORE = 0  # Non-target instances (Weeds in crop-targeted mode)
    MONITOR = 1  # Target detected but outside the active action zone
    ALLOW_ACTION = 2  # Target in action zone with low uncertainty (Ready)
    DENY_ACTION = 3  # Target in action zone but uncertainty is too high (Safety Abort)


# ==============================================================================
# 2. INDIVIDUAL PLANT MEMORY STATE (DATACLASS)
# ==============================================================================

@dataclass
class TrackedPlant:
    """
    Represents a single tracked plant instance over time (Digital Twin).
    """
    plant_id: int
    class_id: int  # 0 for crop, 1 for weed

    # field(default_factory=list) is used to initialize empty lists in dataclasses safely
    entropy_history: List[float] = field(default_factory=list)
    seen_count: int = 0  # Number of consecutive frames detected
    is_stable: bool = False  # True only if seen_count >= min_stable_frames
    smoothed_entropy: float = 1.0  # Starts at maximum uncertainty (1.0)

    # Latest spatial data (for geofencing and visualization)
    bbox: np.ndarray = None  # [x1, y1, x2, y2]
    mask: np.ndarray = None  # Binary mask (bool)

    def update_metrics(self, entropy: float, bbox: np.ndarray, mask: np.ndarray, config: MemoryConfig):
        """
        Updates the plant's history and calculates smoothed temporal uncertainty.
        """
        self.bbox = bbox
        self.mask = mask
        self.seen_count += 1

        # 1. Anti-Flicker check: Has the plant been detected long enough?
        if self.seen_count >= config.min_stable_frames:
            self.is_stable = True

        # 2. Update entropy history (Sliding Window)
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > config.window_size:
            self.entropy_history.pop(0)  # Remove oldest record

        # 3. Calculate smoothed (mean) entropy over the window
        self.smoothed_entropy = float(np.mean(self.entropy_history))


# ==============================================================================
# 3. STATE MANAGER (Manages the active population on the field)
# ==============================================================================

class StateManager:
    """
    Manages the lifecycle of all active tracked plants in the camera frame.
    Ensures zero-memory leaks by deleting lost tracks immediately.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        # Dictionary mapping plant_id -> TrackedPlant object
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
        Updates active plants with new detections, registers new ones, 
        and removes plants that have left the frame.
        """
        current_frame_ids = set()

        # Process new detections
        if len(ids) > 0:
            for i, plant_id in enumerate(ids):
                current_frame_ids.add(plant_id)

                # Calculate raw Shannon Entropy for this single frame detection
                conf = confs[i]
                p_win = np.clip(conf, 1e-5, 1.0 - 1e-5)
                p_lose = 1.0 - p_win
                probs = np.array([p_win, p_lose])
                probs = probs / np.sum(probs)
                raw_entropy = -np.sum(probs * np.log2(probs))

                # Extract spatial data
                bbox = boxes[i]
                mask = masks[i]
                class_id = int(classes[i])

                # If plant is new, register it in memory
                if plant_id not in self.active_plants:
                    self.active_plants[plant_id] = TrackedPlant(
                        plant_id=plant_id,
                        class_id=class_id
                    )

                # Update the plant's state history
                self.active_plants[plant_id].update_metrics(
                    entropy=raw_entropy,
                    bbox=bbox,
                    mask=mask,
                    config=self.config
                )

        # MEMORY CLEANUP: Remove lost plants that left the camera frame
        lost_ids = set(self.active_plants.keys()) - current_frame_ids
        for lost_id in lost_ids:
            del self.active_plants[lost_id]