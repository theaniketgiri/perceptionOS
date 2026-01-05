"""
Configuration management for Perception OS
"""

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class CameraConfig:
    """Camera input configuration"""
    source: int | str = 0  # 0 for webcam, or RTSP URL
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class ModelConfig:
    """Model configuration"""
    detector: str = "yolov8n.pt"  # YOLO model
    detector_confidence: float = 0.65  # Raised from 0.5 to reduce false positives
    embedding_dim: int = 512
    device: str = "cuda"  # or "cpu"


@dataclass
class TrackerConfig:
    """Tracking configuration"""
    max_age: int = 60  # Frames to keep lost track (was 30)
    min_hits: int = 3  # Minimum detections to confirm track
    iou_threshold: float = 0.3
    embedding_threshold: float = 0.6  # For re-identification (was 0.7)
    
    # State transitions - tuned for better persistence
    occlusion_threshold_ms: int = 1000   # 1s before marking occluded (was 500ms)
    uncertain_threshold_ms: int = 3000   # 3s before marking uncertain (was 2s)
    lost_threshold_ms: int = 8000        # 8s before marking lost (was 5s)


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    """Main configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("CAMERA_SOURCE"):
            source = os.getenv("CAMERA_SOURCE")
            config.camera.source = int(source) if source.isdigit() else source
        
        if os.getenv("DEVICE"):
            config.model.device = os.getenv("DEVICE")
        
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))
        
        return config


# Global config instance
config = Config.from_env()
