# decision/policy.py
from typing import Dict
from utils.config_manager import SystemConfig
from decision.state import TrackedPlant, InterventionState


# ==============================================================================
# DECISION ENGINE (POLICY & GEOFENCING - CROP-TARGETED)
# ==============================================================================

class DecisionEngine:
    """
    Evaluates tracked plants and determines robotic actions (Enums)
    targeted specifically at Crops (Class 0). Weeds (Class 1) are ignored.
    """

    def __init__(self, config: SystemConfig):
        self.entropy_threshold = config.decision.entropy_threshold
        self.action_zone_ratio = config.decision.action_zone_ratio

    def evaluate_plants(
            self,
            active_plants: Dict[int, TrackedPlant],
            frame_height: int
    ) -> Dict[int, InterventionState]:
        """
        Evaluates each stable plant and returns its current InterventionState.
        Targets Crops (class 0) for action; ignores Weeds (class 1).
        """
        decisions = {}
        # Define the Y-coordinate boundary for the active action zone
        action_line_y = int(frame_height * (1.0 - self.action_zone_ratio))

        for plant_id, plant in active_plants.items():
            # Only evaluate plants that have stabilized (anti-flicker)
            if not plant.is_stable:
                continue

            # Weeds (class_id == 1) are ignored (always mapped to IGNORE)
            if plant.class_id == 1:
                decisions[plant_id] = InterventionState.IGNORE
                continue

            # Crops (class_id == 0) are the targets and require evaluation
            bbox = plant.bbox
            center_y = int((bbox[1] + bbox[3]) / 2)  # Y-center of bounding box

            # 1. Geofencing check: Is the crop in the active action zone?
            if center_y < action_line_y:
                decisions[plant_id] = InterventionState.MONITOR
            else:
                # 2. Uncertainty check: Is the smoothed entropy below the threshold?
                if plant.smoothed_entropy < self.entropy_threshold:
                    decisions[plant_id] = InterventionState.ALLOW_ACTION
                else:
                    decisions[plant_id] = InterventionState.DENY_ACTION

        return decisions