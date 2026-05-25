import cv2
import numpy as np
from typing import Union
from utils.config_manager import SystemConfig, ClassConfig


# ==============================================================================
# VISUALIZATION UTILS (HIGH-PERFORMANCE RENDERER)
# ==============================================================================

def draw_predictions(
        frame: np.ndarray,
        ids: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        classes: np.ndarray,
        config: SystemConfig
) -> np.ndarray:
    """
    Draws tracking bounding boxes, IDs, and aggregated colored masks on the frame.
    Optimized for single-pass alpha blending and high-throughput execution.
    """
    # 4. Typing & Empty Check: Immediate return to prevent useless copying
    if len(ids) == 0:
        return frame

    annotated_frame = frame.copy()

    # 1. Performance Optimization: Initialize a single overlay canvas for all masks
    # We will accumulate all masks here and blend ONLY ONCE at the end.
    mask_overlay = np.zeros_like(annotated_frame)
    has_masks = masks is not None and len(masks) > 0

    # 3. Production Safety: Determine the minimum safe length to prevent IndexErrors
    n = min(len(ids), len(boxes), len(classes))
    if has_masks:
        n = min(n, len(masks))

    # Core Drawing Loop
    for i in range(n):
        plant_id = int(ids[i])
        x1, y1, x2, y2 = map(int, boxes[i])
        cls_id = int(classes[i])

        # Safe lookup for classes configuration
        if cls_id in config.classes:
            class_cfg: ClassConfig = config.classes[cls_id]
            color = tuple(class_cfg.color)
            label_name = class_cfg.name
        else:
            color = (128, 128, 128)  # Gray fallback
            label_name = f"Unknown_{cls_id}"

        # Draw bounding box and ID label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{label_name} ID: {plant_id}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 2. Mask Safety & Accumulation
        if has_masks:
            # Force explicit boolean type to prevent OpenCV indexing anomalies
            mask = masks[i].astype(bool)
            # Accumulate this plant's mask onto the single overlay canvas
            mask_overlay[mask] = color

    # 1. Single-Pass Blending: Blend the accumulated overlay ONCE (O(1) instead of O(N))
    if has_masks:
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.4, 0)

    return annotated_frame