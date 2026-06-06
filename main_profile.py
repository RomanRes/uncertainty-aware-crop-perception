# main_profile.py
import cv2
import time
from tqdm import tqdm
from utils.config_manager import load_config
from perception.tracker import PlantTracker
from decision.state import StateManager
from decision.policy import DecisionEngine
from utils.logging import SystemLogger
from utils.viz_profile import draw_predictions, viz_step_times
from utils.gpu_helper import GPUHelper
import torch
import sys


def main():
    """
    Main function for running the Industrial Perception Pipeline with detailed profiling.
    It initializes the system, processes video frames, tracks plants,
    makes decisions, logs telemetry, and prints performance summaries.
    """
    print("PROFILING RUN")
    print("MAIN CUDA:", torch.cuda.is_available())
    print("MAIN GPU COUNT:", torch.cuda.device_count())
    print("PYTHON:", sys.executable)

    print("Starting Industrial Perception Pipeline with Profiling...")

    # Initialize profiling accumulators for the main pipeline
    step_times = {
        "video_read": [],
        "perception_inference": [],
        "state_memory_update": [],
        "decision_engine": [],
        "gpu_telemetry": [],
        "rendering_overlays": [],
        "video_write": [],
        "telemetry_logging": []
    }

    # 1. Load System Configuration from YAML into typed Dataclasses
    cfg = load_config("configs/default.yaml")

    # 2. Initialize GPU Telemetry Helper
    gpu_helper = GPUHelper()

    # 3. Initialize Core Modules
    tracker = PlantTracker(cfg)
    state_manager = StateManager(cfg.memory)
    decision_engine = DecisionEngine(cfg)

    # 4. Dynamic Telemetry Path Generation & Logger Initialization
    csv_path = cfg.io.metrics_output_csv
    json_path = csv_path.replace(".csv", ".json")
    logger = SystemLogger(csv_path, json_path)

    # 5. Initialize Video Input
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

    # 6. Initialize Video Output (Supports dynamic width allocation)
    output_width = width * 2 if cfg.io.side_by_side else width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg.io.output_video, fourcc, fps, (output_width, height))

    processed_count = 0

    # ==========================================================================
    # CORE PIPELINE EXECUTION LOOP WITH PROFILING
    # ==========================================================================
    try:
        for frame_idx in tqdm(range(frames_to_process)):
            start_frame_time = time.time()

            # Measure Video Read Time
            t_start = time.time()
            ret, frame = cap.read()
            step_times["video_read"].append((time.time() - t_start) * 1000.0)

            if not ret:
                break

            # STEP 1: Perception (Measure model inference time precisely)
            t_start = time.time()
            ids, boxes, masks, confs, classes = tracker.track_frame(frame)
            inference_time_ms = (time.time() - t_start) * 1000.0
            step_times["perception_inference"].append(inference_time_ms)

            # STEP 2: State Memory Update (Registers and updates stable histories)
            t_start = time.time()
            state_manager.update_state(ids, classes, confs, boxes, masks)
            step_times["state_memory_update"].append((time.time() - t_start) * 1000.0)

            # STEP 3: Decision Engine (Evaluate crop states and dynamic ROI geofencing)
            t_start = time.time()
            decisions = decision_engine.evaluate_plants(state_manager.active_plants, width, height)
            step_times["decision_engine"].append((time.time() - t_start) * 1000.0)

            # STEP 4: Query Real-time GPU Utilization
            t_start = time.time()
            gpu_util = gpu_helper.get_utilization()
            step_times["gpu_telemetry"].append((time.time() - t_start) * 1000.0)

            # Calculate total frame latency for visualization
            frame_time_ms = (time.time() - start_frame_time) * 1000.0
            """
            # STEP 6: Rendering (Draws decision-state-aware overlays with telemetry)
            t_start = time.time()
            annotated_frame = draw_predictions(
                frame=frame,
                active_plants=state_manager.active_plants,
                decisions=decisions,
                config=cfg,
                inference_time_ms=inference_time_ms,
                frame_time_ms=frame_time_ms,
                gpu_util=gpu_util
            )
            step_times["rendering_overlays"].append((time.time() - t_start) * 1000.0)

            # STEP 7: Write output stream
            t_start = time.time()
            out.write(annotated_frame)
            step_times["video_write"].append((time.time() - t_start) * 1000.0)
            """
            # STEP 8: Telemetry Logging (Log all performance metrics to CSV)
            t_start = time.time()
            logger.log_frame(
                frame_idx=frame_idx,
                inference_time_ms=inference_time_ms,
                frame_time_ms=frame_time_ms,
                gpu_util=gpu_util,
                active_plants=state_manager.active_plants,
                decisions=decisions
            )
            step_times["telemetry_logging"].append((time.time() - t_start) * 1000.0)

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

        # --- Print Main Profiling Summary ---
        if processed_count > 0:
            print("\n" + "=" * 83)
            print("PIPELINE STEP PROFILING SUMMARY (Average execution time per frame)")
            print("=" * 83)
            print(f"{'Pipeline Step':<35} {'Avg Time (ms)':>20} {'Percentage (%)':>20}")
            print("-" * 83)

            avg_times = {step: sum(times) / len(times) for step, times in step_times.items() if times}
            total_avg_time = sum(avg_times.values())

            for step, avg_time in avg_times.items():
                percentage = (avg_time / total_avg_time) * 100 if total_avg_time > 0 else 0.0
                print(f"{step:<35} {avg_time:>18.2f} ms {percentage:>18.1f}%")

            print("-" * 83)
            print(f"{'Total Pipeline Latency':<35} {total_avg_time:>18.2f} ms {100.0:>18.1f}%")
            print(f"{'Calculated Throughput':<35} {1000.0 / total_avg_time:>18.1f} FPS")
            print("=" * 83)

        # --- Print Detailed Visualization Profiling Summary ---
        if processed_count > 0 and any(viz_step_times.values()):
            print("\n" + "=" * 83)
            print("DETAILED VISUALIZATION PROFILING SUMMARY (Inside rendering_overlays)")
            print("=" * 83)
            print(f"{'Visualization Sub-Step':<35} {'Avg Time (ms)':>20} {'Percentage (%)':>20}")
            print("-" * 83)

            avg_viz_times = {step: sum(times) / len(times) for step, times in viz_step_times.items() if times}
            total_avg_viz = sum(avg_viz_times.values())

            for step, avg_time in avg_viz_times.items():
                percentage = (avg_time / total_avg_viz) * 100 if total_avg_viz > 0 else 0.0
                print(f"{step:<35} {avg_time:>18.2f} ms {percentage:>18.1f}%")

            print("-" * 83)
            print(f"{'Total Visualization Latency':<35} {total_avg_viz:>18.2f} ms {100.0:>18.1f}%")
            print("=" * 83)

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()