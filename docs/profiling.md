
Test Setup

- GPU: NVIDIA T4
- Resolution: 960×960
- Model: YOLO26s-Seg
- Precision: FP16 (.engine)
- Tracker: ByteTrack
===================================================================================
PIPELINE STEP PROFILING SUMMARY (Average execution time per frame)
===================================================================================
Pipeline Step                              Avg Time (ms)       Percentage (%)
-----------------------------------------------------------------------------------
video_read                                        1.11 ms                2.6%
perception_inference                             39.90 ms               93.3%
state_memory_update                               0.11 ms                0.3%
decision_engine                                   0.03 ms                0.1%
gpu_telemetry                                     1.48 ms                3.5%
telemetry_logging                                 0.14 ms                0.3%
-----------------------------------------------------------------------------------
Total Pipeline Latency                           42.78 ms              100.0%
Calculated Throughput                             23.4 FPS
===================================================================================
Pipeline execution completed successfully.

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