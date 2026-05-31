import pynvml

class GPUHelper:
    """
    Interfaces with NVIDIA Management Library (NVML) to query real-time GPU utilization.
    Provides safe fallback on CPU-only machines.
    """
    def __init__(self):
        self.initialized = False
        try:
            pynvml.nvmlInit()
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.initialized = True
            print("Info: NVML successfully initialized for GPU telemetry.")
        except Exception:
            print("Warning: NVML initialization failed. GPU telemetry falls back to 0%.")

    def get_utilization(self) -> int:
        """
        Returns the current GPU utilization as an integer percentage (0-100).
        """
        if not self.initialized:
            return 0
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle)
            return int(rates.gpu)
        except Exception:
            return 0

    def __del__(self):
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass