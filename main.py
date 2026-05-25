# main.py
import cv2
from tqdm import tqdm
from utils.config_manager import load_config
from perception.tracker import PlantTracker
from utils.viz import draw_predictions


def main():
    print("🎬 Starting High-Performance Agri-Perception Pipeline...")

    # 1. Load System Configuration from YAML into typed Dataclasses
    cfg = load_config("configs/default.yaml")

    # 2. Initialize the optimized Plant Tracker (YOLO + ByteTrack)
    tracker = PlantTracker(cfg)

    # 3. Initialize Video Input
    cap = cv2.VideoCapture(cfg.io.input_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open input video at {cfg.io.input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    # 4. Determine how many frames to process (respecting our quick test config)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if cfg.io.max_frames > 0:
        frames_to_process = min(cfg.io.max_frames, total_frames)
        print(f"⚠️ QUICK TEST ACTIVE: Processing only first {frames_to_process} frames.")
    else:
        frames_to_process = total_frames
        print(f"🎥 Processing full video: {frames_to_process} frames total.")

    # Initialize Video Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg.io.output_video, fourcc, fps, (width, height))

    # ==========================================================================
    # CORE PIPELINE LOOP
    # ==========================================================================
    try:
        # Loop only over the specified number of frames
        for _ in tqdm(range(frames_to_process)):
            ret, frame = cap.read()
            if not ret:
                break

            ids, boxes, masks, confs, classes = tracker.track_frame(frame)
            annotated_frame = draw_predictions(frame, ids, boxes, masks, classes, cfg)
            out.write(annotated_frame)

    except Exception as e:
        print(f"\n❌ Pipeline crashed during processing: {e}")
    finally:
        # Guarantee that resources are freed even if the pipeline crashes
        cap.release()
        out.release()
        print("\n🔒 Video resources released.")

    print(f"🎉 Pipeline successfully finished!")


if __name__ == "__main__":
    main()