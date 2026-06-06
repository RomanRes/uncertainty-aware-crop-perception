# tests/test_state.py
import pytest
import numpy as np
from decision.state import TrackedPlant, StateManager, InterventionState
from utils.config_manager import MemoryConfig


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def memory_config():
    """Fixture for a default MemoryConfig instance."""
    return MemoryConfig(window_size=5, min_stable_frames=3)


@pytest.fixture
def state_manager(memory_config):
    """Fixture for a StateManager instance with a default MemoryConfig."""
    return StateManager(config=memory_config, max_missing_frames=3)


def make_dummy_bbox():
    """Creates a dummy bounding box for testing."""
    return np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)


def make_dummy_mask(h=480, w=640):
    """Creates a dummy segmentation mask for testing."""
    mask = np.zeros((h, w), dtype=bool)
    mask[100:200, 100:200] = True
    return mask


def make_frame_data(n=1, class_id=0, conf=0.9):
    """
    Generates dummy frame data for testing.

    Args:
        n (int): Number of detections to generate.
        class_id (int): Class ID for the detections.
        conf (float): Confidence score for the detections.

    Returns:
        Tuple: ids, classes, confs, boxes, masks
    """
    ids     = np.array([i + 1 for i in range(n)], dtype=np.int32)
    classes = np.full(n, class_id, dtype=np.int32)
    confs   = np.full(n, conf, dtype=np.float32)
    boxes   = np.tile(make_dummy_bbox(), (n, 1))
    masks   = np.array([make_dummy_mask() for _ in range(n)], dtype=bool)
    return ids, classes, confs, boxes, masks


# ==============================================================================
# TrackedPlant Tests
# ==============================================================================

class TestTrackedPlant:
    """Tests for the TrackedPlant dataclass."""

    def test_initial_state(self):
        """A new plant should not be stable and have 0 confidence."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        assert plant.is_stable is False
        assert plant.seen_count == 0
        assert plant.smoothed_conf == 0.0
        assert plant.bbox is None

    def test_becomes_stable_after_min_frames(self, memory_config):
        """Plant becomes stable after min_stable_frames updates."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        bbox = make_dummy_bbox()
        mask = make_dummy_mask()

        for i in range(memory_config.min_stable_frames - 1):
            plant.update_history(0.9, bbox, mask, memory_config)
            assert plant.is_stable is False

        plant.update_history(0.9, bbox, mask, memory_config)
        assert plant.is_stable is True

    def test_smoothed_conf_is_mean_of_history(self, memory_config):
        """smoothed_conf should be the average of the confidence history."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        confs = [0.9, 0.8, 0.7]
        for c in confs:
            plant.update_history(c, make_dummy_bbox(), make_dummy_mask(), memory_config)

        assert abs(plant.smoothed_conf - np.mean(confs)) < 1e-5

    def test_sliding_window_max_size(self, memory_config):
        """Confidence history should not exceed window_size."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        for i in range(memory_config.window_size + 3):
            plant.update_history(float(i) / 10.0, make_dummy_bbox(), make_dummy_mask(), memory_config)

        assert len(plant.conf_history) <= memory_config.window_size

    def test_missing_count_resets_on_update(self, memory_config):
        """missing_count should reset to 0 when the plant reappears."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        plant.missing_count = 5
        plant.update_history(0.9, make_dummy_bbox(), make_dummy_mask(), memory_config)
        assert plant.missing_count == 0

    def test_bbox_updated_on_each_frame(self, memory_config):
        """Bounding box should be overwritten with each update."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        bbox1 = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
        bbox2 = np.array([200.0, 200.0, 300.0, 300.0], dtype=np.float32)

        plant.update_history(0.9, bbox1, make_dummy_mask(), memory_config)
        assert np.allclose(plant.bbox, bbox1)

        plant.update_history(0.9, bbox2, make_dummy_mask(), memory_config)
        assert np.allclose(plant.bbox, bbox2)

    def test_low_confidence_smoothed(self, memory_config):
        """Low confidence should be smoothed correctly."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        for c in [0.1, 0.1, 0.1]:
            plant.update_history(c, make_dummy_bbox(), make_dummy_mask(), memory_config)
        assert plant.smoothed_conf < 0.2


# ==============================================================================
# StateManager Tests
# ==============================================================================

class TestStateManager:
    """Tests for the StateManager class."""

    def test_empty_frame_no_crash(self, state_manager):
        """An empty frame should not cause a crash."""
        ids     = np.empty((0,), dtype=np.int32)
        classes = np.empty((0,), dtype=np.int32)
        confs   = np.empty((0,), dtype=np.float32)
        boxes   = np.empty((0, 4), dtype=np.float32)
        masks   = np.empty((0, 480, 640), dtype=bool)

        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert len(state_manager.active_plants) == 0

    def test_new_plant_registered(self, state_manager):
        """A new detection should be registered as a TrackedPlant."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert 1 in state_manager.active_plants
        assert state_manager.active_plants[1].class_id == 0

    def test_multiple_plants_registered(self, state_manager):
        """Multiple detections should all be registered."""
        ids, classes, confs, boxes, masks = make_frame_data(n=3)
        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert len(state_manager.active_plants) == 3

    def test_grace_period_keeps_plant_alive(self, state_manager):
        """A plant should remain in memory during its grace period."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)

        empty    = np.empty((0,), dtype=np.int32)
        empty_b  = np.empty((0, 4), dtype=np.float32)
        empty_m  = np.empty((0, 480, 640), dtype=bool)

        for _ in range(state_manager.max_missing_frames):
            state_manager.update_state(empty, empty, empty, empty_b, empty_m)
            assert 1 in state_manager.active_plants

    def test_plant_deleted_after_grace_period(self, state_manager):
        """A plant should be deleted after the grace period expires."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)

        empty   = np.empty((0,), dtype=np.int32)
        empty_b = np.empty((0, 4), dtype=np.float32)
        empty_m = np.empty((0, 480, 640), dtype=bool)

        for _ in range(state_manager.max_missing_frames + 1):
            state_manager.update_state(empty, empty, empty, empty_b, empty_m)

        assert 1 not in state_manager.active_plants

    def test_plant_reappears_resets_missing_count(self, state_manager):
        """A reappearing plant should have its missing_count reset to 0."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)

        empty   = np.empty((0,), dtype=np.int32)
        empty_b = np.empty((0, 4), dtype=np.float32)
        empty_m = np.empty((0, 480, 640), dtype=bool)
        state_manager.update_state(empty, empty, empty, empty_b, empty_m)
        state_manager.update_state(empty, empty, empty, empty_b, empty_m)

        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert state_manager.active_plants[1].missing_count == 0

    def test_weed_and_crop_tracked_separately(self, state_manager):
        """Crop (0) and Weed (1) should be tracked independently."""
        ids     = np.array([1, 2], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        confs   = np.array([0.9, 0.8], dtype=np.float32)
        boxes   = np.tile(make_dummy_bbox(), (2, 1))
        masks   = np.array([make_dummy_mask(), make_dummy_mask()], dtype=bool)

        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert state_manager.active_plants[1].class_id == 0
        assert state_manager.active_plants[2].class_id == 1