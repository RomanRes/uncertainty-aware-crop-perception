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
    return MemoryConfig(window_size=5, min_stable_frames=3)


@pytest.fixture
def state_manager(memory_config):
    return StateManager(config=memory_config, max_missing_frames=3)


def make_dummy_bbox():
    return np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)


def make_dummy_mask(h=480, w=640):
    mask = np.zeros((h, w), dtype=bool)
    mask[100:200, 100:200] = True
    return mask


def make_frame_data(n=1, class_id=0, conf=0.9):
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

    def test_initial_state(self):
        """Neue Pflanze ist nicht stabil und hat 0 Confidence."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        assert plant.is_stable is False
        assert plant.seen_count == 0
        assert plant.smoothed_conf == 0.0
        assert plant.bbox is None

    def test_becomes_stable_after_min_frames(self, memory_config):
        """Pflanze wird stabil nach min_stable_frames Updates."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        bbox = make_dummy_bbox()
        mask = make_dummy_mask()

        for i in range(memory_config.min_stable_frames - 1):
            plant.update_history(0.9, bbox, mask, memory_config)
            assert plant.is_stable is False

        plant.update_history(0.9, bbox, mask, memory_config)
        assert plant.is_stable is True

    def test_smoothed_conf_is_mean_of_history(self, memory_config):
        """smoothed_conf ist der Durchschnitt der Konfidenz-Historie."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        confs = [0.9, 0.8, 0.7]
        for c in confs:
            plant.update_history(c, make_dummy_bbox(), make_dummy_mask(), memory_config)

        assert abs(plant.smoothed_conf - np.mean(confs)) < 1e-5

    def test_sliding_window_max_size(self, memory_config):
        """Konfidenz-Historie überschreitet window_size nicht."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        for i in range(memory_config.window_size + 3):
            plant.update_history(float(i) / 10.0, make_dummy_bbox(), make_dummy_mask(), memory_config)

        assert len(plant.conf_history) <= memory_config.window_size

    def test_missing_count_resets_on_update(self, memory_config):
        """missing_count wird auf 0 zurückgesetzt wenn Pflanze wieder auftaucht."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        plant.missing_count = 5
        plant.update_history(0.9, make_dummy_bbox(), make_dummy_mask(), memory_config)
        assert plant.missing_count == 0

    def test_bbox_updated_on_each_frame(self, memory_config):
        """bbox wird bei jedem Update überschrieben."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        bbox1 = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
        bbox2 = np.array([200.0, 200.0, 300.0, 300.0], dtype=np.float32)

        plant.update_history(0.9, bbox1, make_dummy_mask(), memory_config)
        assert np.allclose(plant.bbox, bbox1)

        plant.update_history(0.9, bbox2, make_dummy_mask(), memory_config)
        assert np.allclose(plant.bbox, bbox2)

    def test_low_confidence_smoothed(self, memory_config):
        """Niedrige Konfidenz wird korrekt geglättet."""
        plant = TrackedPlant(plant_id=1, class_id=0)
        for c in [0.1, 0.1, 0.1]:
            plant.update_history(c, make_dummy_bbox(), make_dummy_mask(), memory_config)
        assert plant.smoothed_conf < 0.2


# ==============================================================================
# StateManager Tests
# ==============================================================================

class TestStateManager:

    def test_empty_frame_no_crash(self, state_manager):
        """Leerer Frame führt zu keinem Fehler."""
        ids     = np.empty((0,), dtype=np.int32)
        classes = np.empty((0,), dtype=np.int32)
        confs   = np.empty((0,), dtype=np.float32)
        boxes   = np.empty((0, 4), dtype=np.float32)
        masks   = np.empty((0, 480, 640), dtype=bool)

        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert len(state_manager.active_plants) == 0

    def test_new_plant_registered(self, state_manager):
        """Neue Detektion wird als TrackedPlant registriert."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert 1 in state_manager.active_plants
        assert state_manager.active_plants[1].class_id == 0

    def test_multiple_plants_registered(self, state_manager):
        """Mehrere Detektionen werden alle registriert."""
        ids, classes, confs, boxes, masks = make_frame_data(n=3)
        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert len(state_manager.active_plants) == 3

    def test_grace_period_keeps_plant_alive(self, state_manager):
        """Pflanze bleibt während der Grace Period im Speicher."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)

        empty    = np.empty((0,), dtype=np.int32)
        empty_b  = np.empty((0, 4), dtype=np.float32)
        empty_m  = np.empty((0, 480, 640), dtype=bool)

        for _ in range(state_manager.max_missing_frames):
            state_manager.update_state(empty, empty, empty, empty_b, empty_m)
            assert 1 in state_manager.active_plants

    def test_plant_deleted_after_grace_period(self, state_manager):
        """Pflanze wird nach Ablauf der Grace Period gelöscht."""
        ids, classes, confs, boxes, masks = make_frame_data(n=1)
        state_manager.update_state(ids, classes, confs, boxes, masks)

        empty   = np.empty((0,), dtype=np.int32)
        empty_b = np.empty((0, 4), dtype=np.float32)
        empty_m = np.empty((0, 480, 640), dtype=bool)

        for _ in range(state_manager.max_missing_frames + 1):
            state_manager.update_state(empty, empty, empty, empty_b, empty_m)

        assert 1 not in state_manager.active_plants

    def test_plant_reappears_resets_missing_count(self, state_manager):
        """Pflanze die wieder auftaucht bekommt missing_count = 0."""
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
        """Crop (0) und Weed (1) werden unabhängig getrackt."""
        ids     = np.array([1, 2], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        confs   = np.array([0.9, 0.8], dtype=np.float32)
        boxes   = np.tile(make_dummy_bbox(), (2, 1))
        masks   = np.array([make_dummy_mask(), make_dummy_mask()], dtype=bool)

        state_manager.update_state(ids, classes, confs, boxes, masks)
        assert state_manager.active_plants[1].class_id == 0
        assert state_manager.active_plants[2].class_id == 1