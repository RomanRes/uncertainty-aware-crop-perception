from dataclasses import dataclass
import yaml
import os

@dataclass
class ClassConfig:
    name: str
    color: list # List of 3 ints: [B, G, R]

# ==============================================================================
# SUB-CONFIGURATIONS (NESTED DATACLASSES)
# ==============================================================================

@dataclass
class ModelConfig:
    path: str
    imgsz: int
    conf_threshold: float
    tracker_type: str
    max_missing_frames: int
    edge_margin: int
    ema_alpha: float


@dataclass
class MemoryConfig:
    window_size: int
    min_stable_frames: int


@dataclass
class DecisionConfig:
    entropy_threshold: float
    action_zone_ratio: float


@dataclass
class IOConfig:
    input_video: str
    output_video: str
    metrics_output_csv: str
    metrics_output_json: str
    max_frames: int
    side_by_side: bool
    show_telemetry: bool


# ==============================================================================
# MAIN SYSTEM CONFIGURATION
# ==============================================================================

@dataclass
class SystemConfig:
    model: ModelConfig
    memory: MemoryConfig
    decision: DecisionConfig
    io: IOConfig
    classes: ClassConfig




# ==============================================================================
# CONFIG LOADER FUNCTION
# ==============================================================================

def load_config(config_path: str = "configs/default.yaml") -> SystemConfig:
    """
    Loads a YAML configuration file and parses it into typed Dataclasses.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    classes_dict = {
        int(k): ClassConfig(**v) for k, v in data["classes"].items()
    }

    # Programmatic mapping of dictionary keys to structured Dataclasses
    return SystemConfig(
        model=ModelConfig(**data["model"]),
        memory=MemoryConfig(**data["memory"]),
        decision=DecisionConfig(**data["decision"]),
        io=IOConfig(**data["io"]),
        classes = classes_dict
    )