from ultralytics import YOLO

# 1. Load your locally downloaded PyTorch weights on CPU
# Replace the path with the exact location of your .pt file
model_path = "models/phenobench_cropweed_seg_yolo11s_960.pt"
model = YOLO(model_path)

print("Starting CPU-based ONNX export...")

# 2. Export to ONNX (opset=13 is optimal for C++ compatibility)
# This will run entirely on your CPU and create the .onnx file next to your .pt file
model.export(format="onnx", opset=13, imgsz=960, dynamic=True)

print("ONNX export completed successfully on CPU!")