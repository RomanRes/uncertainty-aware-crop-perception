
Test Setup

- GPU: NVIDIA T4
- Resolution: 960×960
- Model: YOLO11s-Seg
- Precision: FP32
- Tracker: ByteTrack
===================================================================================
PIPELINE STEP PROFILING SUMMARY (Average execution time per frame)
===================================================================================
Pipeline Step                              Avg Time (ms)       Percentage (%)
-----------------------------------------------------------------------------------
video_read                                        1.45 ms                0.6%
perception_inference                            139.20 ms               60.6%
state_memory_update                               0.10 ms                0.0%
decision_engine                                   0.03 ms                0.0%
gpu_telemetry                                     1.52 ms                0.7%
rendering_overlays                               73.36 ms               31.9%
video_write                                      13.88 ms                6.0%
telemetry_logging                                 0.20 ms                0.1%
-----------------------------------------------------------------------------------
Total Pipeline Latency                          229.75 ms              100.0%
Calculated Throughput                              4.4 FPS
===================================================================================

===================================================================================
DETAILED VISUALIZATION PROFILING SUMMARY (Inside rendering_overlays)
===================================================================================
Visualization Sub-Step                     Avg Time (ms)       Percentage (%)
-----------------------------------------------------------------------------------
mask_overlay_init                                 0.25 ms                0.3%
box_and_text_render                              33.07 ms               45.4%
mask_blending_weighted                            0.95 ms                1.3%
uncertainty_canvas_gen                           34.50 ms               47.4%
side_by_side_stacking                             1.25 ms                1.7%
telemetry_overlay                                 2.82 ms                3.9%
-----------------------------------------------------------------------------------
Total Visualization Latency                      72.83 ms              100.0%
===================================================================================