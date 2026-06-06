# tests/test_policy.py
import pytest
import numpy as np
from decision.policy import DecisionEngine
from decision.state import TrackedPlant, InterventionState
from utils.config_manager import SystemConfig, ModelConfig, MemoryConfig, DecisionConfig, IOConfig, ClassConfig


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def config():
    """Fixture for a default SystemConfig instance for testing."""
    return SystemConfig(
        model=ModelConfig(
            path="model.pt",
            imgsz=960,
            conf_threshold=0.5,
            tracker_type="bytetrack.yaml",
            max_missing_frames=10,
            edge_margin=20,
            ema_alpha=0.7
        ),
        memory=MemoryConfig(window_size=5, min_stable_frames=3),
        decision=DecisionConfig(entropy_threshold=0.3, action_zone_ratio=0.5),
        io=IOConfig(
            input_video="video.mp4",
            output_video="out.mp4",
            metrics_output_csv="metrics.csv",
            metrics_output_json="metrics.json",
            max_frames=-1,
            side_by_side=False,
            show_telemetry=False
        ),
        classes={
            0: ClassConfig(name="Crop", color=[144, 238, 144]),
            1: ClassConfig(name="Weed", color=[122, 122, 244])
        }
    )


@pytest.fixture
def engine(config):
    """Fixture for a DecisionEngine instance with the default config."""
    return DecisionEngine(config)


# Image dimensions for all tests
FRAME_W = 640
FRAME_H = 480


def make_stable_plant(plant_id: int, class_id: int, bbox: list, smoothed_conf: float) -> TrackedPlant:
    """
    Helper function to create a stable plant with a given bounding box and confidence.

    Args:
        plant_id (int): The ID of the plant.
        class_id (int): The class ID of the plant.
        bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        smoothed_conf (float): The smoothed confidence score.

    Returns:
        TrackedPlant: A TrackedPlant instance configured as stable.
    """
    plant = TrackedPlant(plant_id=plant_id, class_id=class_id)
    plant.is_stable = True
    plant.bbox = np.array(bbox, dtype=np.float32)
    plant.smoothed_conf = smoothed_conf
    return plant


def center_of(bbox):
    """Calculates the center coordinates of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# ==============================================================================
# Weed Tests
# ==============================================================================

class TestWeedPolicy:
    """Tests related to the policy for weed detection."""

    def test_weed_always_ignored(self, engine):
        """Weeds (class 1) should always be IGNORE, regardless of position and confidence."""
        # In ROI, high confidence - still IGNORE
        plant = make_stable_plant(1, class_id=1, bbox=[200, 300, 400, 450], smoothed_conf=0.99)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.IGNORE

    def test_weed_outside_roi_also_ignored(self, engine):
        """Weeds outside ROI should also be IGNORE."""
        plant = make_stable_plant(1, class_id=1, bbox=[10, 10, 50, 50], smoothed_conf=0.9)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.IGNORE


# ==============================================================================
# Stability Guard Tests
# ==============================================================================

class TestStabilityGuard:
    """Tests related to the stability guard for tracked plants."""

    def test_unstable_plant_skipped(self, engine):
        """An unstable plant should not appear in decisions."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        plant.is_stable = False
        plant.bbox = np.array([200, 300, 400, 450], dtype=np.float32)
        plant.smoothed_conf = 0.9

        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert 1 not in decisions

    def test_plant_without_bbox_skipped(self, engine):
        """A plant without a bounding box should not appear in decisions."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        plant.is_stable = True
        plant.bbox = None
        plant.smoothed_conf = 0.9

        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert 1 not in decisions


# ==============================================================================
# ROI Geofencing Tests
# ==============================================================================

class TestROIGeofencing:
    """
    Tests related to Region of Interest (ROI) geofencing.
    With action_zone_ratio=0.5:
    ROI Y: 240 to 456 (for H=480)
    ROI X: 32 to 608 (for W=640)
    """

    def test_crop_inside_roi_high_conf_allow(self, engine):
        """Crop inside ROI with high confidence should result in ALLOW_ACTION."""
        # Center of the image = center of ROI
        plant = make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.9)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.ALLOW_ACTION

    def test_crop_inside_roi_low_conf_deny(self, engine):
        """Crop inside ROI with low confidence should result in DENY_ACTION."""
        plant = make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.1)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.DENY_ACTION

    def test_crop_above_roi_monitor(self, engine):
        """Crop above the ROI should result in MONITOR."""
        # Y-center at 50 -> far above ROI boundary (240)
        plant = make_stable_plant(1, class_id=0, bbox=[200, 10, 400, 90], smoothed_conf=0.9)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.MONITOR

    def test_conf_exactly_at_threshold_allow(self, engine):
        """Confidence exactly at threshold should result in ALLOW_ACTION (>=)."""
        plant = make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.5)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.ALLOW_ACTION

    def test_conf_just_below_threshold_deny(self, engine):
        """Confidence just below threshold should result in DENY_ACTION."""
        plant = make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.49)
        decisions = engine.evaluate_plants({1: plant}, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.DENY_ACTION


# ==============================================================================
# Multi-Plant Tests
# ==============================================================================

class TestMultiPlant:
    """Tests for scenarios involving multiple plants."""

    def test_empty_plants_returns_empty_decisions(self, engine):
        """No plants should result in empty decisions."""
        decisions = engine.evaluate_plants({}, FRAME_W, FRAME_H)
        assert decisions == {}

    def test_mixed_plants_correct_decisions(self, engine):
        """Mixed plants (crop in ROI, weed, crop outside ROI) should all be handled correctly."""
        plants = {
            1: make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.9),  # ALLOW
            2: make_stable_plant(2, class_id=1, bbox=[270, 280, 370, 380], smoothed_conf=0.9),  # IGNORE
            3: make_stable_plant(3, class_id=0, bbox=[200, 10,  400, 90],  smoothed_conf=0.9),  # MONITOR
        }
        decisions = engine.evaluate_plants(plants, FRAME_W, FRAME_H)

        assert decisions[1] == InterventionState.ALLOW_ACTION
        assert decisions[2] == InterventionState.IGNORE
        assert decisions[3] == InterventionState.MONITOR

    def test_two_crops_different_confidence(self, engine):
        """Two crops in ROI with different confidences should result in one ALLOW and one DENY."""
        plants = {
            1: make_stable_plant(1, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.9),
            2: make_stable_plant(2, class_id=0, bbox=[270, 280, 370, 380], smoothed_conf=0.1),
        }
        decisions = engine.evaluate_plants(plants, FRAME_W, FRAME_H)
        assert decisions[1] == InterventionState.ALLOW_ACTION
        assert decisions[2] == InterventionState.DENY_ACTION