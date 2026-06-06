# utils/viz.py
import cv2
import numpy as np
from typing import Dict
from utils.config_manager import SystemConfig, ClassConfig
from decision.state import TrackedPlant, InterventionState


def _generate_uncertainty_canvas(
        frame: np.ndarray,
        active_plants: Dict[int, TrackedPlant],
        config: SystemConfig,
        action_line_y: int
) -> np.ndarray:
    """
    Generates a canvas visualizing uncertainty for active plants.

    Args:
        frame (np.ndarray): The original video frame.
        active_plants (Dict[int, TrackedPlant]): Dictionary of currently tracked plants.
        config (SystemConfig): The system configuration.
        action_line_y (int): The y-coordinate of the action line.

    Returns:
        np.ndarray: The generated uncertainty canvas.
    """
    h, w, _ = frame.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.line(canvas, (0, action_line_y), (w, action_line_y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "ACTION LIMIT", (10, action_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    mask_overlay = np.zeros_like(canvas)
    has_masks = False

    for plant_id, plant in active_plants.items():
        if not plant.is_stable or plant.bbox is None:
            continue

        x1, y1, x2, y2 = map(int, plant.bbox)
        cls_id = plant.class_id

        label_name = config.classes[cls_id].name if cls_id in config.classes else f"Unknown_{cls_id}"

        conf = float(np.clip(plant.smoothed_conf, 0.0, 1.0))
        color = (0, int(255 * conf), int(255 * (1.0 - conf)))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
        label = f"{label_name} #{plant_id} | C:{plant.smoothed_conf:.2f}"
        cv2.putText(canvas, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1, cv2.LINE_AA)

        if plant.mask is not None:
            has_masks = True
            mask_overlay[plant.mask] = color

    if has_masks:
        canvas = cv2.addWeighted(canvas, 1.0, mask_overlay, 0.5, 0)

    return canvas


# ==============================================================================
# VISUALIZATION UTILS
# ==============================================================================

def draw_predictions(
        frame: np.ndarray,
        active_plants: Dict[int, TrackedPlant],
        decisions: Dict[int, InterventionState],
        config: SystemConfig,
        inference_time_ms: float = 0.0,
        frame_time_ms: float = 0.0,
        gpu_util: int = 0
) -> np.ndarray:
    """
    Draws predictions, bounding boxes, masks, and telemetry onto the video frame.

    Args:
        frame (np.ndarray): The original video frame.
        active_plants (Dict[int, TrackedPlant]): Dictionary of currently tracked plants.
        decisions (Dict[int, InterventionState]): Dictionary of intervention decisions for each plant.
        config (SystemConfig): The system configuration.
        inference_time_ms (float): Time taken for model inference in milliseconds.
        frame_time_ms (float): Total time taken to process the current frame in milliseconds.
        gpu_util (int): Current GPU utilization percentage.

    Returns:
        np.ndarray: The annotated frame with predictions and telemetry.
    """

    h, w, _ = frame.shape
    action_line_y = int(h * (1.0 - config.decision.action_zone_ratio))

    # Calculate telemetry values once
    fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0

    # --- EDGE CASE: no plants ---
    if not active_plants:
        if config.io.side_by_side:
            combined_frame = np.hstack((frame, np.zeros_like(frame)))
        else:
            combined_frame = frame.copy()

        if config.io.show_telemetry:
            _draw_telemetry(combined_frame, fps, inference_time_ms, gpu_util)

        return combined_frame

    annotated_frame = frame.copy()

    cv2.line(annotated_frame, (0, action_line_y), (w, action_line_y), (255, 255, 255), 2)
    cv2.putText(annotated_frame, "ACTION ZONE LIMIT", (10, action_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # A single mask_overlay for all masks
    mask_overlay = np.zeros_like(annotated_frame)
    has_masks = False

    for plant_id, plant in active_plants.items():
        if not plant.is_stable or plant.bbox is None:
            continue

        x1, y1, x2, y2 = map(int, plant.bbox)
        cls_id = plant.class_id

        state = decisions.get(plant_id, InterventionState.MONITOR)

        if state == InterventionState.IGNORE:
            color = (128, 128, 128)
            label_text = f"Weed #{plant_id} (IGNORE)"
        elif state == InterventionState.MONITOR:
            color = (255, 255, 0)
            label_text = f"Crop #{plant_id} (MONITOR) | C: {plant.smoothed_conf:.2f}"
        elif state == InterventionState.ALLOW_ACTION:
            color = tuple(config.classes[cls_id].color) if cls_id in config.classes else (0, 255, 0)
            label_text = f"Crop #{plant_id} (ALLOW_ACTION) | C: {plant.smoothed_conf:.2f}"
        elif state == InterventionState.DENY_ACTION:
            color = (0, 0, 255)
            label_text = f"Crop #{plant_id} (DENY_ACTION) | C: {plant.smoothed_conf:.2f}"
        else:
            color = (128, 128, 128)
            label_text = f"#{plant_id}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        if plant.mask is not None:
            has_masks = True
            mask_overlay[plant.mask] = color

    if has_masks:
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.4, 0)

    # --- SIDE-BY-SIDE ---
    if config.io.side_by_side:
        try:
            uncertainty_frame = _generate_uncertainty_canvas(frame, active_plants, config, action_line_y)
            if not isinstance(uncertainty_frame, np.ndarray) or uncertainty_frame.ndim != 3:
                raise ValueError("Invalid Canvas")
        except Exception as e:
            print(f"\nWarning: Uncertainty rendering failed: {e}. Using black fallback.")
            uncertainty_frame = np.zeros_like(annotated_frame)

        combined_frame = np.hstack((annotated_frame, uncertainty_frame))
    else:
        combined_frame = annotated_frame

    # --- TELEMETRY ---
    if config.io.show_telemetry:
        _draw_telemetry(combined_frame, fps, inference_time_ms, gpu_util)

    return combined_frame


def _draw_telemetry(frame: np.ndarray, fps: float, inference_ms: float, gpu_util: int):
    """
    Draws system telemetry information (FPS, inference time, GPU utilization) onto the frame.

    Args:
        frame (np.ndarray): The frame to draw telemetry on.
        fps (float): Frames per second.
        inference_ms (float): Inference time in milliseconds.
        gpu_util (int): GPU utilization percentage.
    """
    box_w, box_h = 220, 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "System Telemetry",          (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}",           (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (0, 255, 0),    1, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {inference_ms:.1f} ms", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"GPU Util: {gpu_util}%",    (15, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 0),  1, cv2.LINE_AA)