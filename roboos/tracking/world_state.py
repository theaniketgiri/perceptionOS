"""
World state manager - Central state for the perception system.
Aggregates tracking data and provides a unified view.
"""

import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from loguru import logger

from roboos.perception.camera import Frame
from roboos.perception.detector import Detection, Detector
from roboos.tracking.tracker import Tracker, TrackedObject


@dataclass
class WorldState:
    """Complete snapshot of the world state"""
    timestamp_ms: int
    frame_id: int
    objects: List[dict]
    summary: dict
    
    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "objects": self.objects,
        }


class WorldStateManager:
    """
    Central manager for world state.
    Coordinates perception and tracking modules.
    """
    
    def __init__(self):
        self.detector = Detector()
        self.tracker = Tracker()
        
        self._current_state: Optional[WorldState] = None
        self._frame_count = 0
        self._start_time_ms = int(time.time() * 1000)
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing World State Manager")
        
        if not self.detector.load():
            logger.error("Failed to load detector")
            return False
        
        logger.info("World State Manager initialized")
        return True
    
    def process_frame(self, frame: Frame) -> WorldState:
        """
        Process a single frame and update world state.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Updated world state
        """
        self._frame_count += 1
        
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(
            detections=detections,
            timestamp_ms=frame.timestamp_ms,
            frame_id=frame.frame_id
        )
        
        # Build world state
        objects = [obj.to_dict() for obj in tracked_objects]
        summary = self.tracker.get_summary()
        
        self._current_state = WorldState(
            timestamp_ms=frame.timestamp_ms,
            frame_id=frame.frame_id,
            objects=objects,
            summary=summary
        )
        
        # Update FPS counter
        self._update_fps()
        
        return self._current_state
    
    def _update_fps(self):
        """Update FPS calculation"""
        self._fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = current_time
    
    def get_world_state(self) -> Optional[dict]:
        """Get current world state as dictionary"""
        if self._current_state is None:
            return None
        return self._current_state.to_dict()
    
    def get_summary(self) -> dict:
        """Get summary of world state"""
        return self.tracker.get_summary()
    
    def get_object(self, obj_id: str) -> Optional[dict]:
        """Get a single object by ID"""
        obj = self.tracker.get_object(obj_id)
        if obj is None:
            return None
        return obj.to_dict()
    
    def get_health(self) -> dict:
        """Get system health status"""
        import torch
        
        gpu_name = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        
        return {
            "status": "ok",
            "fps": round(self._fps, 1),
            "frame_count": self._frame_count,
            "gpu": gpu_name,
            "objects_tracked": len(self.tracker.objects)
        }
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
