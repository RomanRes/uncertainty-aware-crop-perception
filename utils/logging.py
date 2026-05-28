# utils/logging.py
import os
import csv
import json
import time
import numpy as np
from typing import Dict, List, Set
from decision.state import TrackedPlant, InterventionState


# ==============================================================================
# INDUSTRIAL PERFORMANCE & TELEMETRY LOGGER
# ==============================================================================

class SystemLogger:
    """
    Logs frame-level telemetry to CSV and compiles a comprehensive JSON summary
    at the end of the pipeline execution. Tracks latency, tracking stability, and decisions.
    """

    def __init__(self, csv_path: str, json_path: str):
        self.csv_path = csv_path
        self.json_path = json_path

        # Performance accumulator
        self.frame_times: List[float] = []
        self.inference_times: List[float] = []

        # Tracking telemetry
        self.active_lifetimes: Dict[int, int] = {}  # plant_id -> active frame count
        self.completed_lifetimes: List[int] = []  # List of lifetimes of deleted tracks
        self.tracked_classes: Dict[int, int] = {}  # plant_id -> class_id (to detect class swaps)
        self.id_swaps = 0

        # Decision counters
        self.total_allow_action = 0
        self.total_deny_action = 0
        self.total_monitor = 0

        self._initialize_csv()

    def _initialize_csv(self):
        # Only create directory if a directory path is actually specified
        dir_name = os.path.dirname(self.csv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx",
                "inference_time_ms",
                "frame_time_ms",
                "fps",
                "active_tracks",
                "allow_action_count",
                "deny_action_count"
            ])

    def log_frame(
            self,
            frame_idx: int,
            inference_time_ms: float,
            frame_time_ms: float,
            active_plants: Dict[int, TrackedPlant],
            decisions: Dict[int, InterventionState]
    ):
        """
        Logs telemetry for a single frame and updates internal metrics.
        """
        self.frame_times.append(frame_time_ms)
        self.inference_times.append(inference_time_ms)

        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
        active_tracks = len(active_plants)

        # 1. Update Tracking Lifetimes and detect Class Swaps (ID Switches)
        current_ids = set(active_plants.keys())
        for plant_id, plant in active_plants.items():
            # Update lifetime
            if plant_id not in self.active_lifetimes:
                self.active_lifetimes[plant_id] = 0
                self.tracked_classes[plant_id] = plant.class_id
            else:
                self.active_lifetimes[plant_id] += 1
                # Check for ID swaps (class mismatch for same tracking ID)
                if self.tracked_classes[plant_id] != plant.class_id:
                    self.id_swaps += 1
                    self.tracked_classes[plant_id] = plant.class_id  # Update class mapping

        # Archive lifetimes of plants that left the frame
        lost_ids = set(self.active_lifetimes.keys()) - current_ids
        for lost_id in lost_ids:
            self.completed_lifetimes.append(self.active_lifetimes[lost_id])
            del self.active_lifetimes[lost_id]
            del self.tracked_classes[lost_id]

        # 2. Count Decisions
        allow_count = sum(1 for act in decisions.values() if act == InterventionState.ALLOW_ACTION)
        deny_count = sum(1 for act in decisions.values() if act == InterventionState.DENY_ACTION)
        monitor_count = sum(1 for act in decisions.values() if act == InterventionState.MONITOR)

        self.total_allow_action += allow_count
        self.total_deny_action += deny_count
        self.total_monitor += monitor_count

        # 3. Write Row to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_idx,
                f"{inference_time_ms:.2f}",
                f"{frame_time_ms:.2f}",
                f"{fps:.1f}",
                active_tracks,
                allow_count,
                deny_count
            ])

    def save_summary_json(self, total_frames: int):
        """
        Compiles and saves a comprehensive JSON report summarizing the entire run.
        """
        # Append remaining active plant lifetimes to completed list
        all_lifetimes = self.completed_lifetimes + list(self.active_lifetimes.values())
        mean_lifetime = float(np.mean(all_lifetimes)) if all_lifetimes else 0.0
        max_lifetime = int(np.max(all_lifetimes)) if all_lifetimes else 0

        mean_inference_time = float(np.mean(self.inference_times)) if self.inference_times else 0.0
        mean_frame_time = float(np.mean(self.frame_times)) if self.frame_times else 0.0
        mean_fps = 1000.0 / mean_frame_time if mean_frame_time > 0 else 0.0

        summary = {
            "metadata": {
                "total_frames_processed": total_frames,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_metrics": {
                "mean_inference_time_ms": round(mean_inference_time, 2),
                "mean_frame_time_ms": round(mean_frame_time, 2),
                "mean_fps": round(mean_fps, 1)
            },
            "tracking_metrics": {
                "total_unique_ids_tracked": len(all_lifetimes) + len(self.completed_lifetimes),
                "mean_track_lifetime_frames": round(mean_lifetime, 1),
                "max_track_lifetime_frames": max_lifetime,
                "detected_id_class_swaps": self.id_swaps
            },
            "decision_metrics": {
                "total_allow_action_events": self.total_allow_action,
                "total_deny_action_events": self.total_deny_action,
                "total_monitor_events": self.total_monitor
            }
        }

        # Save JSON
        dir_name = os.path.dirname(self.json_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"📊 JSON Summary successfully compiled and saved to: {self.json_path}")