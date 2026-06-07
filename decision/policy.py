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
        self.conf_threshold = config.model.conf_threshold
        self.action_zone_ratio = config.decision.action_zone_ratio
        self.entropy_threshold = config.decision.entropy_threshold

    def evaluate_plants(
            self,
            active_plants: Dict[int, TrackedPlant],
            frame_width: int,
            frame_height: int
    ) -> Dict[int, InterventionState]:
        """
        Evaluates active plants against the ROI and confidence constraints,
        implementing a decision locking mechanism once a plant enters the action zone.
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
                decisions[plant_id] = InterventionState.MONITOR # Default to monitor if not stable or no bbox
                plant.is_in_action_zone = False
                plant.decision_locked = False
                continue

            # Non-targets (Weeds = Class 1) are strictly ignored (IGNORE)
            if plant.class_id == 1:
                decisions[plant_id] = InterventionState.IGNORE
                plant.is_in_action_zone = False
                plant.decision_locked = False
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

            if is_inside_roi:
                # Plant is in the action zone
                if not plant.is_in_action_zone:
                    # First time entering action zone: lock its state for decision-making
                    plant.is_in_action_zone = True
                    plant.decision_locked = True # This prevents further updates to smoothed_conf/entropy in TrackedPlant.update_history

                    # Make the decision based on the current (now locked) entropy
                    if plant.entropy > self.entropy_threshold:
                        decisions[plant_id] = InterventionState.DENY_ACTION
                    else:
                        decisions[plant_id] = InterventionState.ALLOW_ACTION
                else:
                    # Plant is already in the action zone and decision is locked
                    # Re-evaluate decision based on its *locked* entropy (which hasn't changed)
                    if plant.entropy > self.entropy_threshold:
                        decisions[plant_id] = InterventionState.DENY_ACTION
                    else:
                        decisions[plant_id] = InterventionState.ALLOW_ACTION
            else:
                # Plant is outside the action zone
                plant.is_in_action_zone = False
                plant.decision_locked = False # Unlock its state
                decisions[plant_id] = InterventionState.MONITOR

        return decisions