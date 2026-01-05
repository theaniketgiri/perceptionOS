"""
Camera input handler for Perception OS. Supports webcam and RTSP streams.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from dataclasses import dataclass
import time
from loguru import logger

from roboos.config import config


@dataclass
class Frame:
    """A single camera frame with metadata"""
    image: np.ndarray
    timestamp_ms: int
    frame_id: int
    width: int
    height: int


class Camera:
    """Camera input handler with buffering and FPS control"""
    
    def __init__(
        self,
        source: int | str = None,
        width: int = None,
        height: int = None,
        fps: int = None
    ):
        self.source = source if source is not None else config.camera.source
        self.width = width or config.camera.width
        self.height = height or config.camera.height
        self.target_fps = fps or config.camera.fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._start_time_ms = 0
        
    def open(self) -> bool:
        """Open the camera connection"""
        logger.info(f"Opening camera source: {self.source}")
        
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self.source}")
            return False
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual values
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        self._start_time_ms = int(time.time() * 1000)
        self._frame_id = 0
        
        return True
    
    def close(self):
        """Close the camera connection"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed")
    
    def read(self) -> Optional[Frame]:
        """Read a single frame from the camera"""
        if self._cap is None or not self._cap.isOpened():
            return None
        
        ret, image = self._cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            return None
        
        self._frame_id += 1
        timestamp_ms = int(time.time() * 1000) - self._start_time_ms
        
        return Frame(
            image=image,
            timestamp_ms=timestamp_ms,
            frame_id=self._frame_id,
            width=image.shape[1],
            height=image.shape[0]
        )
    
    def stream(self) -> Generator[Frame, None, None]:
        """Generate a stream of frames"""
        if not self.open():
            return
        
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()
    
    def __enter__(self) -> "Camera":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def preprocess_frame(frame: Frame, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess frame for model input.
    
    Args:
        frame: Input frame
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image as numpy array (RGB, normalized)
    """
    image = frame.image
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize with aspect ratio preservation
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_w = target_size[0] - new_w
    pad_h = target_size[1] - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image
