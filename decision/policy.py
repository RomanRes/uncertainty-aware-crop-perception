# decision/policy.py
from typing import Dict
import numpy as np
from utils.config_manager import SystemConfig
from decision.state import TrackedPlant, InterventionState


# ==============================================================================
# DECISION ENGINE (ROI GEOFENCING & SYSTEM POLICY)
# ==============================================================================

class DecisionEngine:
    """
    Evaluates tracked plants and determines active robotic states
    using a dynamic Region of Interest (ROI) and smoothed confidences.
    """

    def __init__(self, config: SystemConfig):
        # We now use the smoothed confidence score directly
        self.conf_threshold = config.model.conf_threshold
        self.action_zone_ratio = config.decision.action_zone_ratio

    def evaluate_plants(
            self,
            active_plants: Dict[int, TrackedPlant],
            frame_width: int,
            frame_height: int
    ) -> Dict[int, InterventionState]:
        """
        Evaluates active plants against the ROI and confidence constraints.
        """
        decisions = {}

        # 1. Define a dynamic ROI (Region of Interest) box in the center-lower frame
        roi_y_min = int(frame_height * (1.0 - self.action_zone_ratio))
        roi_y_max = int(frame_height * 0.95)  # Avoid absolute image edge
        roi_x_min = int(frame_width * 0.05)
        roi_x_max = int(frame_width * 0.95)

        for plant_id, plant in active_plants.items():
            # Anti-Flicker & Crash Safety: Ensure stable tracking and valid bounding boxes
            if not plant.is_stable or plant.bbox is None:
                continue

            # Non-targets (Weeds = Class 1) are strictly ignored (IGNORE)
            if plant.class_id == 1:
                decisions[plant_id] = InterventionState.IGNORE
                continue

            # Crops (Class 0) are the targets and require evaluation
            x1, y1, x2, y2 = plant.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 2. Dynamic ROI Check (Geofencing)
            is_inside_roi = (
                    roi_y_min <= center_y <= roi_y_max and
                    roi_x_min <= center_x <= roi_x_max
            )

            if not is_inside_roi:
                decisions[plant_id] = InterventionState.MONITOR
            else:
                # 3. Decision based on smoothed temporal confidence
                if plant.smoothed_conf >= self.conf_threshold:
                    decisions[plant_id] = InterventionState.ALLOW_ACTION
                else:
                    decisions[plant_id] = InterventionState.DENY_ACTION

        return decisions