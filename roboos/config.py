"""
Configuration management for Perception OS.

Design Principles:
    1. Single source of truth for all configurable parameters
    2. Environment variable overrides for deployment flexibility
    3. Sensible defaults that work out of the box
    4. Type-safe dataclasses with validation

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, FrozenSet, Optional
from enum import Enum
import os


# =============================================================================
# Class Categories - COCO class IDs organized by semantic type
# =============================================================================

class ObjectCategory(str, Enum):
    """Semantic categories for tracked objects."""
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    FURNITURE = "furniture"
    ELECTRONICS = "electronics"
    OTHER = "other"


# COCO class ID mappings (YOLOv8 uses COCO classes)
# Reference: https://docs.ultralytics.com/datasets/detect/coco/
COCO_CLASSES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

# Classes that are typically dynamic and worth tracking
DYNAMIC_CLASSES: FrozenSet[int] = frozenset({
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    6,   # train
    7,   # truck
    8,   # boat
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
})

# Static/furniture classes to ignore by default
STATIC_CLASSES: FrozenSet[int] = frozenset({
    13,  # bench
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    61,  # toilet
    62,  # tv
    68,  # microwave
    69,  # oven
    70,  # toaster
    71,  # sink
    72,  # refrigerator
})


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class CameraConfig:
    """Camera input configuration."""
    source: int | str = 0  # 0 for webcam, or RTSP URL
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class DetectionConfig:
    """Object detection configuration."""
    model: str = "yolov8n.pt"
    confidence: float = 0.65
    embedding_dim: int = 512
    device: str = "cuda"
    
    # Class filtering
    filter_enabled: bool = True
    allowed_classes: Optional[FrozenSet[int]] = None  # None = use DYNAMIC_CLASSES
    
    def __post_init__(self):
        if self.allowed_classes is None:
            self.allowed_classes = DYNAMIC_CLASSES
    
    def is_class_allowed(self, class_id: int) -> bool:
        """Check if a class ID should be tracked."""
        if not self.filter_enabled:
            return True
        return class_id in self.allowed_classes
    
    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name."""
        return COCO_CLASSES.get(class_id, f"class_{class_id}")


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration."""
    max_age: int = 60           # Frames to keep lost track
    min_hits: int = 3           # Minimum detections to confirm track
    iou_threshold: float = 0.3
    embedding_threshold: float = 0.6
    
    # State transition thresholds (milliseconds)
    occlusion_threshold_ms: int = 1000   # Before marking occluded
    uncertain_threshold_ms: int = 3000   # Before marking uncertain
    lost_threshold_ms: int = 8000        # Before marking lost


@dataclass
class EncoderConfig:
    """Semantic encoder configuration."""
    # Encoder type: "histogram" (fast, real-time) or "vjepa" (slow, better quality)
    # Default: histogram for real-time FPS on RTX 3050
    encoder_type: str = "histogram"
    
    # V-JEPA model variants:
    # - "facebook/vjepa2-vitl-fpc64-256" (large, recommended)
    # - "facebook/vjepa2-vitg-fpc64-384-ssv2" (giant, best quality)
    vjepa_model: str = "facebook/vjepa2-vitl-fpc64-256"
    
    # Embedding dimension (auto-detected from model)
    embedding_dim: int = 384
    
    # Device for encoder inference
    device: str = "cuda"


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    """
    Main configuration container.
    
    Usage:
        from roboos.config import config
        
        if config.detection.is_class_allowed(class_id):
            # track this object
    """
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Backwards compatibility aliases
    @property
    def model(self) -> DetectionConfig:
        return self.detection
    
    @classmethod
    def from_env(cls) -> Config:
        """
        Load configuration with environment variable overrides.
        
        Supported env vars:
            CAMERA_SOURCE: Camera index or RTSP URL
            DEVICE: 'cuda' or 'cpu'
            API_PORT: Server port number
            FILTER_CLASSES: 'true' or 'false'
        """
        config = cls()
        
        if os.getenv("CAMERA_SOURCE"):
            source = os.getenv("CAMERA_SOURCE")
            config.camera.source = int(source) if source.isdigit() else source
        
        if os.getenv("DEVICE"):
            config.detection.device = os.getenv("DEVICE")
        
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))
        
        if os.getenv("FILTER_CLASSES"):
            config.detection.filter_enabled = os.getenv("FILTER_CLASSES").lower() == "true"
        
        if os.getenv("DETECTION_CONFIDENCE"):
            config.detection.confidence = float(os.getenv("DETECTION_CONFIDENCE"))
        
        if os.getenv("ENCODER_TYPE"):
            config.encoder.encoder_type = os.getenv("ENCODER_TYPE").lower()
        
        return config


# =============================================================================
# Global Config Instance
# =============================================================================

config = Config.from_env()


# =============================================================================
# Utility Functions
# =============================================================================

def get_allowed_class_names() -> list[str]:
    """Get list of class names that will be tracked."""
    return [
        COCO_CLASSES[class_id] 
        for class_id in sorted(config.detection.allowed_classes)
    ]


def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 50)
    print("Perception OS Configuration")
    print("=" * 50)
    print(f"Camera: {config.camera.source} @ {config.camera.width}x{config.camera.height}")
    print(f"Detection: {config.detection.model}, conf={config.detection.confidence}")
    print(f"Device: {config.detection.device}")
    print(f"Class Filter: {'enabled' if config.detection.filter_enabled else 'disabled'}")
    if config.detection.filter_enabled:
        print(f"Tracking classes: {get_allowed_class_names()}")
    print(f"API: http://{config.api.host}:{config.api.port}")
    print("=" * 50)
