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
    Helper: Generates the black-background uncertainty canvas.
    Displays dynamic class names, IDs, bounding boxes, and uncertainty-glowing masks.
    """
    h, w, _ = frame.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw geofencing limit on the black canvas
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

        # Dynamic Class Name lookup from the YAML configuration
        if cls_id in config.classes:
            label_name = config.classes[cls_id].name
        else:
            label_name = f"Unknown_{cls_id}"

        # Linear BGR interpolation based on smoothed confidence (Green = Certain, Red = Uncertain)
        conf = np.clip(plant.smoothed_conf, 0.0, 1.0)
        green_val = int(255 * conf)
        red_val = int(255 * (1.0 - conf))
        color = (0, green_val, red_val)

        # Draw neutral white bounding boxes
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # NEW: Draw dynamic Class Name, ID, and Smoothed Confidence
        label = f"{label_name} #{plant_id} | C:{plant.smoothed_conf:.2f}"
        cv2.putText(canvas, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1, cv2.LINE_AA)

        if plant.mask is not None:
            has_masks = True
            mask = plant.mask.astype(bool)
            mask_overlay[mask] = color

            # Das addWeighted wird ausgeführt
    if has_masks:
        canvas = cv2.addWeighted(canvas, 1.0, mask_overlay, 0.5, 0)

            # CRITICAL FIX: Return the fully rendered 3D canvas back to the caller!
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
    Draws stabilized bounding boxes, IDs, and aggregated masks on the frame.
    Colors and labels are dynamically adjusted based on the active InterventionState.
    """
    h, w, _ = frame.shape
    action_line_y = int(h * (1.0 - config.decision.action_zone_ratio))

    # --- EDGE CASE SAFETY ---
    # If no plants are currently in memory, we must still respect the side_by_side setting
    # to prevent shape mismatches in the OpenCV VideoWriter.
    if not active_plants:
        if config.io.side_by_side:
            blank_uncertainty = np.zeros_like(frame)
            combined_frame = np.hstack((frame, blank_uncertainty))
        else:
            combined_frame = frame.copy()

        # Draw only the telemetry if enabled, even on empty frames
        if config.io.show_telemetry:
            fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
            box_w, box_h = 220, 90
            overlay = combined_frame.copy()
            cv2.rectangle(overlay, (5, 5), (box_w, box_h), (0, 0, 0), -1)
            combined_frame = cv2.addWeighted(overlay, 0.5, combined_frame, 0.5, 0)

            cv2.putText(combined_frame, "System Telemetry", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined_frame, f"FPS: {fps:.1f}", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(combined_frame, f"Inference: {inference_time_ms:.1f} ms", (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined_frame, f"GPU Util: {gpu_util}%", (15, 79),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        return combined_frame

    annotated_frame = frame.copy()

    # 1. Geofencing: Calculate and draw the active action zone limit
    cv2.line(annotated_frame, (0, action_line_y), (w, action_line_y), (255, 255, 255), 2)
    cv2.putText(annotated_frame, "ACTION ZONE LIMIT", (10, action_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Initialize a single overlay canvas for all masks (Single-pass blending)
    mask_overlay = np.zeros_like(annotated_frame)
    has_masks = False

    # 2. Iterate through all active plants in memory
    for plant_id, plant in active_plants.items():
        # Anti-Flicker: Only render plants that have stabilized
        if not plant.is_stable or plant.bbox is None:
            continue

        x1, y1, x2, y2 = map(int, plant.bbox)
        cls_id = plant.class_id

        # Get the computed decision state (default to MONITOR if not evaluated)
        state = decisions.get(plant_id, InterventionState.MONITOR)

        # 3. Dynamic Color and Label Mapping based on InterventionState
        if state == InterventionState.IGNORE:
            color = (128, 128, 128)
            label_text = f"Weed #{plant_id} (IGNORE)"

        elif state == InterventionState.MONITOR:
            color = (255, 255, 0)
            label_text = f"Crop #{plant_id} (MONITOR) | C: {plant.smoothed_conf:.2f}"

        elif state == InterventionState.ALLOW_ACTION:
            if cls_id in config.classes:
                color = tuple(config.classes[cls_id].color)
            else:
                color = (0, 255, 0)
            label_text = f"Crop #{plant_id} (ALLOW_ACTION) | C: {plant.smoothed_conf:.2f}"

        elif state == InterventionState.DENY_ACTION:
            color = (0, 0, 255)
            label_text = f"Crop #{plant_id} (DENY_ACTION) | C: {plant.smoothed_conf:.2f}"

        # 4. Draw Bounding Box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # 5. Draw ID and State Label
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 6. Accumulate Segmentation Mask
        if plant.mask is not None:
            has_masks = True
            mask = plant.mask.astype(bool)
            mask_overlay[mask] = color

    # 7. Single-Pass Blending: Blend the accumulated overlay once (O(1))
    if has_masks:
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.4, 0)

    # --- PART 3: HORIZONTAL CONCATENATION (SIDE-BY-SIDE) ---
    if config.io.side_by_side:
        try:
            uncertainty_frame = _generate_uncertainty_canvas(frame, active_plants, config, action_line_y)
            if uncertainty_frame is None or not isinstance(uncertainty_frame, np.ndarray) or len(
                    uncertainty_frame.shape) != 3:
                raise ValueError(f"Invalid uncertainty canvas returned. Type: {type(uncertainty_frame)}")
        except Exception as e:
            print(f"\nWarning: Uncertainty rendering failed: {e}. Using black fallback.")
            uncertainty_frame = np.zeros_like(annotated_frame)

        combined_frame = np.hstack((annotated_frame, uncertainty_frame))
        # REMOVED: No early return here!
    else:
        combined_frame = annotated_frame

    # --- PART 4: SYSTEM TELEMETRY OVERLAY ---
    if config.io.show_telemetry:
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0

        # Draw a semi-transparent black background box in the top-left corner
        box_w, box_h = 220, 90
        overlay = combined_frame.copy()
        cv2.rectangle(overlay, (5, 5), (box_w, box_h), (0, 0, 0), -1)
        combined_frame = cv2.addWeighted(overlay, 0.5, combined_frame, 0.5, 0)

        # Render the text parameters
        cv2.putText(combined_frame, "System Telemetry", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, f"FPS: {fps:.1f}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, f"Inference: {inference_time_ms:.1f} ms", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, f"GPU Util: {gpu_util}%", (15, 79),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # SINGLE RETURN STATEMENT
    return combined_frame