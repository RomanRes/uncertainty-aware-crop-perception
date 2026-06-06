# tests/test_tracker.py
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from utils.config_manager import SystemConfig, ModelConfig, MemoryConfig, DecisionConfig, IOConfig, ClassConfig


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def config():
    """Fixture for a default SystemConfig instance for testing."""
    return SystemConfig(
        model=ModelConfig(
            path="models/model.pt",
            imgsz=960,
            conf_threshold=0.15,
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


def make_dummy_frame(h=480, w=640):
    """Creates a dummy image frame for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_mock_results(n=2, h=480, w=640, has_masks=True):
    """
    Creates a mock Ultralytics Results object with 'n' detections for testing.
    """
    results = MagicMock()

    import torch
    results.boxes.id   = torch.tensor([float(i + 1) for i in range(n)])
    results.boxes.xyxy = torch.tensor([[50.0, 50.0, 150.0, 150.0] for _ in range(n)])
    results.boxes.conf = torch.tensor([0.9] * n)
    results.boxes.cls  = torch.tensor([0.0] * n)
    results.boxes.__len__ = lambda self: n

    if has_masks:
        mask_data = torch.zeros((n, h // 4, w // 4))
        mask_data[:, 10:20, 10:20] = 1.0
        results.masks = MagicMock()
        results.masks.data = mask_data
    else:
        results.masks = None

    return results


# ==============================================================================
# Output Format Tests (no real model required)
# ==============================================================================

class TestTrackerOutputFormat:
    """
    Verifies the output structure of track_frame() using a mocked YOLO model.
    No GPU or actual YOLO weights are required for these tests.
    """

    def _make_tracker(self, config):
        """Instantiates PlantTracker with a fully mocked YOLO backend."""
        with patch("perception.tracker.YOLO") as MockYOLO, \
             patch("perception.tracker.torch.cuda.is_available", return_value=False), \
             patch("perception.tracker.os.path.exists", return_value=False):

            mock_model = MagicMock()
            MockYOLO.return_value = mock_model

            from perception.tracker import PlantTracker
            tracker = PlantTracker(config)
            tracker.model = mock_model
            return tracker

    def test_empty_frame_returns_correct_shapes(self, config):
        """When the model returns no detections, all output arrays have zero length."""
        tracker = self._make_tracker(config)
        frame = make_dummy_frame()
        h, w = frame.shape[:2]

        mock_results = MagicMock()
        mock_results.boxes = None
        tracker.model.track.return_value = [mock_results]

        ids, boxes, masks, confs, classes = tracker.track_frame(frame)

        assert ids.shape     == (0,)
        assert boxes.shape   == (0, 4)
        assert masks.shape   == (0, h, w)
        assert confs.shape   == (0,)
        assert classes.shape == (0,)

    def test_none_tracking_ids_returns_empty(self, config):
        """When boxes.id is None, the tracker must return empty arrays."""
        tracker = self._make_tracker(config)
        frame = make_dummy_frame()

        mock_results = MagicMock()
        mock_results.boxes.id = None
        mock_results.boxes.__len__ = lambda self: 1
        tracker.model.track.return_value = [mock_results]

        ids, boxes, masks, confs, classes = tracker.track_frame(frame)
        assert len(ids) == 0

    def test_output_arrays_have_correct_dtypes(self, config):
        """All output arrays must have their expected numpy dtypes."""
        tracker = self._make_tracker(config)
        frame = make_dummy_frame(h=480, w=640)

        mock_results = make_mock_results(n=2, h=480, w=640)
        tracker.model.track.return_value = [mock_results]

        ids, boxes, masks, confs, classes = tracker.track_frame(frame)

        if len(ids) > 0:
            assert ids.dtype     == np.int32
            assert boxes.dtype   == np.float32
            assert masks.dtype   == bool
            assert confs.dtype   == np.float32
            assert classes.dtype == np.int32

    def test_all_output_arrays_have_same_length(self, config):
        """ids, boxes, masks, confs, and classes must all have the same length N."""
        tracker = self._make_tracker(config)
        frame = make_dummy_frame(h=480, w=640)

        mock_results = make_mock_results(n=2, h=480, w=640)
        tracker.model.track.return_value = [mock_results]

        ids, boxes, masks, confs, classes = tracker.track_frame(frame)

        n = len(ids)
        assert len(boxes)   == n
        assert len(confs)   == n
        assert len(classes) == n
        if n > 0:
            assert masks.shape[0] == n


# ==============================================================================
# EMA Smoothing Tests
# ==============================================================================

class TestEMASmoothing:
    """Tests for Exponential Moving Average (EMA) smoothing of bounding boxes."""

    def _make_tracker_with_mock(self, config):
        """Helper to create a PlantTracker instance with a mocked YOLO backend."""
        with patch("perception.tracker.YOLO") as MockYOLO, \
             patch("perception.tracker.torch.cuda.is_available", return_value=False), \
             patch("perception.tracker.os.path.exists", return_value=False):

            mock_model = MagicMock()
            MockYOLO.return_value = mock_model

            from perception.tracker import PlantTracker
            tracker = PlantTracker(config)
            tracker.model = mock_model
            return tracker

    def test_first_detection_returns_raw_box(self, config):
        """The first detection for an ID has no history, so it returns the raw box unchanged."""
        tracker = self._make_tracker_with_mock(config)
        box = np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float32)

        tracker.smoothed_boxes = {}
        smooth = tracker._apply_ema(1, box)
        assert np.allclose(smooth, box)

    def test_ema_blends_toward_new_box(self, config):
        """EMA output should be a weighted blend: alpha * new + (1 - alpha) * previous."""
        tracker = self._make_tracker_with_mock(config)

        prev_box = np.array([0.0,   0.0,   100.0, 100.0], dtype=np.float32)
        new_box  = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)

        tracker.smoothed_boxes[1] = prev_box
        smooth = tracker._apply_ema(1, new_box)

        # alpha=0.7: expected = 0.7 * new + 0.3 * prev
        expected = 0.7 * new_box + 0.3 * prev_box
        assert np.allclose(smooth, expected, atol=1e-4)


# ==============================================================================
# Edge Filter Tests
# ==============================================================================

class TestEdgeFilter:
    """Tests for filtering detections near image edges."""

    def _make_tracker_with_mock(self, config):
        """Helper to create a PlantTracker instance with a mocked YOLO backend."""
        with patch("perception.tracker.YOLO") as MockYOLO, \
             patch("perception.tracker.torch.cuda.is_available", return_value=False), \
             patch("perception.tracker.os.path.exists", return_value=False):

            mock_model = MagicMock()
            MockYOLO.return_value = mock_model

            from perception.tracker import PlantTracker
            tracker = PlantTracker(config)
            tracker.model = mock_model
            return tracker

    def test_detection_at_top_edge_is_filtered(self, config):
        """A detection whose y1 is within the top edge margin must be filtered out."""
        tracker = self._make_tracker_with_mock(config)
        # y1=5 < edge_margin=20 -> filtered
        assert tracker._is_at_edge(y1=5.0, y2=100.0, h=480, edge_margin=20) is True

    def test_detection_at_bottom_edge_is_filtered(self, config):
        """A detection whose y2 exceeds the bottom edge margin must be filtered out."""
        tracker = self._make_tracker_with_mock(config)
        # y2=470 > h - edge_margin = 460 -> filtered
        assert tracker._is_at_edge(y1=200.0, y2=470.0, h=480, edge_margin=20) is True

    def test_detection_in_center_is_not_filtered(self, config):
        """A detection fully within the safe image area must not be filtered."""
        tracker = self._make_tracker_with_mock(config)
        assert tracker._is_at_edge(y1=100.0, y2=300.0, h=480, edge_margin=20) is False