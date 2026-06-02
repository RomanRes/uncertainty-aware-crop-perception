# decision/state.py
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Deque, Dict
from utils.config_manager import MemoryConfig


class InterventionState(Enum):
    IGNORE       = 0
    MONITOR      = 1
    ALLOW_ACTION = 2
    DENY_ACTION  = 3


@dataclass
class TrackedPlant:
    plant_id: int
    class_id: int

    # OPT 6: deque mit maxlen statt list + manuelles pop(0)
    #         deque.append ist O(1), list.pop(0) ist O(N)
    conf_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    seen_count:   int   = 0
    missing_count: int  = 0
    is_stable:    bool  = False
    smoothed_conf: float = 0.0

    bbox: np.ndarray = None
    mask: np.ndarray = None

    def update_history(self, conf: float, bbox: np.ndarray, mask: np.ndarray, config: MemoryConfig):
        self.bbox = bbox
        self.mask = mask
        self.seen_count  += 1
        self.missing_count = 0

        if self.seen_count >= config.min_stable_frames:
            self.is_stable = True

        # OPT 6: maxlen auf window_size setzen damit kein manuelles pop nötig
        if self.conf_history.maxlen != config.window_size:
            self.conf_history = deque(self.conf_history, maxlen=config.window_size)

        self.conf_history.append(conf)

        # OPT 7: mean direkt mit sum/len statt np.mean für kleine Listen
        self.smoothed_conf = sum(self.conf_history) / len(self.conf_history)


class StateManager:

    def __init__(self, config: MemoryConfig, max_missing_frames: int = 5):
        self.config = config
        self.max_missing_frames = max_missing_frames
        self.active_plants: Dict[int, TrackedPlant] = {}

    def update_state(self, ids, classes, confs, boxes, masks):
        current_frame_ids = self._update_active_tracks(ids, classes, confs, boxes, masks)
        self._cleanup_lost_tracks(current_frame_ids)

    def _update_active_tracks(self, ids, classes, confs, boxes, masks) -> set:
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
        lost_ids = set(self.active_plants.keys()) - current_frame_ids

        for lost_id in list(lost_ids):
            plant = self.active_plants[lost_id]
            plant.missing_count += 1

            if plant.missing_count > self.max_missing_frames:
                del self.active_plants[lost_id]