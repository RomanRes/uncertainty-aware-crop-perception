# main.py
import cv2
import time
from tqdm import tqdm
from utils.config_manager import load_config
from perception.tracker import PlantTracker
from decision.state import StateManager
from decision.policy import DecisionEngine
from utils.logging import SystemLogger
from utils.viz import draw_predictions


def main():
    print("Starting Industrial Perception Pipeline...")

    # 1. Load System Configuration from YAML into typed Dataclasses
    cfg = load_config("configs/default.yaml")

    # 2. Initialize Core Modules
    tracker = PlantTracker(cfg)
    state_manager = StateManager(cfg.memory)
    decision_engine = DecisionEngine(cfg)

    # 3. Dynamic Telemetry Path Generation & Logger Initialization
    csv_path = cfg.io.metrics_output_csv
    json_path = csv_path.replace(".csv", ".json")
    logger = SystemLogger(csv_path, json_path)

    # 4. Initialize Video Input
    cap = cv2.VideoCapture(cfg.io.input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open input video at {cfg.io.input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frames to process (respecting quick test configuration)
    if cfg.io.max_frames > 0:
        frames_to_process = min(cfg.io.max_frames, total_frames)
        print(f"DEBUG: Processing limited to first {frames_to_process} frames.")
    else:
        frames_to_process = total_frames
        print(f"Processing full video: {frames_to_process} frames.")

    # 5. Initialize Video Output

    output_width = width * 2 if cfg.io.side_by_side else width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg.io.output_video, fourcc, fps, (output_width, height))

    processed_count = 0

    # ==========================================================================
    # CORE PIPELINE EXECUTION LOOP
    # ==========================================================================
    try:
        for frame_idx in tqdm(range(frames_to_process)):
            # Start timer for the entire frame pipeline latency
            start_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # STEP 1: Perception (Measure model inference time precisely)
            start_inf_time = time.time()
            ids, boxes, masks, confs, classes = tracker.track_frame(frame)
            inference_time_ms = (time.time() - start_inf_time) * 1000.0

            # STEP 2: State Memory Update (Registers and updates stable histories)
            state_manager.update_state(ids, classes, confs, boxes, masks)

            # STEP 3: Decision Engine (Evaluate crop states and dynamic ROI geofencing)
            decisions = decision_engine.evaluate_plants(state_manager.active_plants, width, height)

            # STEP 4: Rendering (Draws decision-state-aware overlays with 4 arguments)
            annotated_frame = draw_predictions(frame, state_manager.active_plants, decisions, cfg)

            # STEP 5: Write output stream
            out.write(annotated_frame)

            # STEP 6: Telemetry Logging (Calculate total frame latency and log)
            frame_time_ms = (time.time() - start_frame_time) * 1000.0
            logger.log_frame(
                frame_idx=frame_idx,
                inference_time_ms=inference_time_ms,
                frame_time_ms=frame_time_ms,
                active_plants=state_manager.active_plants,
                decisions=decisions
            )

            processed_count += 1

    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        raise e
    finally:
        # Guarantee resources are freed and final JSON report is saved even on failure
        cap.release()
        out.release()

        # Save the structured run summary
        if processed_count > 0:
            logger.save_summary_json(total_frames=processed_count)

        print("\nResources released.")

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()