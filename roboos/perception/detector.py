"""
Object detector using YOLO for bounding box detection.

Responsibilities:
    1. Run YOLO inference on camera frames
    2. Filter detections by class (only track dynamic objects)
    3. Use semantic encoder for embedding extraction

Design Notes:
    - Detection filtering happens at this layer, not in tracker
    - Embeddings use pluggable encoder (V-JEPA or histogram fallback)
    - All CUDA operations happen here

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Final
from dataclasses import dataclass
import torch
from ultralytics import YOLO
from loguru import logger

from roboos.config import config
from roboos.perception.encoder import (
    Encoder, create_encoder, EncoderConfig, EncoderType,
    HistogramEncoder, VJEPAEncoder
)


# =============================================================================
# Constants
# =============================================================================

MIN_CROP_SIZE: Final[int] = 10  # Minimum crop size in pixels


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class Detection:
    """
    A single detected object.
    
    Invariants:
        - bbox is [x1, y1, x2, y2] in pixel coordinates
        - class_id is COCO class ID
        - confidence is in [0, 1]
        - embedding is normalized feature vector
    """
    bbox: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    embedding: Optional[np.ndarray] = None
    
    @property
    def center(self) -> np.ndarray:
        """Center point [x, y] of bounding box."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ], dtype=np.float32)
    
    @property
    def width(self) -> float:
        """Width of bounding box in pixels."""
        return float(self.bbox[2] - self.bbox[0])
    
    @property
    def height(self) -> float:
        """Height of bounding box in pixels."""
        return float(self.bbox[3] - self.bbox[1])
    
    @property
    def area(self) -> float:
        """Area of bounding box in square pixels."""
        return self.width * self.height
    
    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"center=[{self.center[0]:.0f}, {self.center[1]:.0f}])"
        )


# =============================================================================
# Detector Implementation
# =============================================================================

class Detector:
    """
    Object detector using YOLOv8 with semantic encoder.
    
    Features:
        - CUDA-accelerated inference
        - Configurable class filtering (only track dynamic objects)
        - Pluggable encoder: V-JEPA for semantic embeddings, histogram fallback
    
    Usage:
        detector = Detector()
        detector.load()
        detections = detector.detect(frame)  # Already filtered with embeddings
    """
    
    __slots__ = (
        'model_path', 'confidence_threshold', 'device',
        '_model', '_class_names', '_filter_enabled', '_encoder'
    )
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize detector with configuration.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum detection confidence
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path or config.detection.model
        self.confidence_threshold = confidence_threshold or config.detection.confidence
        self.device = device or config.detection.device
        
        self._model: Optional[YOLO] = None
        self._class_names: dict[int, str] = {}
        self._filter_enabled = config.detection.filter_enabled
        self._encoder: Optional[Encoder] = None
    
    def load(self) -> bool:
        """
        Load the YOLO model and encoder onto the specified device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load YOLO detector
            logger.info(f"Loading detector model: {self.model_path}")
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            self._class_names = self._model.names
            
            logger.info(f"Detector loaded on {self.device}, classes: {len(self._class_names)}")
            
            if self._filter_enabled:
                from roboos.config import get_allowed_class_names
                allowed = get_allowed_class_names()
                logger.info(f"Class filter enabled, tracking: {allowed}")
            
            # Load semantic encoder
            self._load_encoder()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return False
    
    def _load_encoder(self) -> None:
        """Load the semantic encoder based on configuration."""
        encoder_type = config.encoder.encoder_type.lower()
        
        try:
            if encoder_type == "vjepa":
                self._encoder = VJEPAEncoder(
                    model_name=config.encoder.vjepa_model,
                    device=config.encoder.device
                )
                if not self._encoder.load():
                    raise RuntimeError("V-JEPA load failed")
                logger.info(f"V-JEPA encoder loaded: {config.encoder.vjepa_model}")
                
            elif encoder_type == "histogram":
                self._encoder = HistogramEncoder(
                    embedding_dim=config.encoder.embedding_dim
                )
                self._encoder.load()
                logger.info("Histogram encoder loaded")
                
            else:  # auto mode
                logger.info("Encoder: auto mode, trying V-JEPA...")
                try:
                    self._encoder = VJEPAEncoder(
                        model_name=config.encoder.vjepa_model,
                        device=config.encoder.device
                    )
                    if self._encoder.load():
                        logger.info(f"V-JEPA encoder loaded: {config.encoder.vjepa_model}")
                    else:
                        raise RuntimeError("V-JEPA load returned False")
                except Exception as e:
                    logger.warning(f"V-JEPA unavailable ({e}), using histogram encoder")
                    self._encoder = HistogramEncoder(
                        embedding_dim=config.encoder.embedding_dim
                    )
                    self._encoder.load()
                    
        except Exception as e:
            logger.error(f"Encoder load failed: {e}, using histogram fallback")
            self._encoder = HistogramEncoder(embedding_dim=config.encoder.embedding_dim)
            self._encoder.load()
    
    def detect(self, frame) -> List[Detection]:
        """
        Detect and filter objects in a frame.
        
        Args:
            frame: Camera frame with .image attribute (BGR numpy array)
            
        Returns:
            List of Detection objects (already filtered by class)
        """
        if self._model is None:
            logger.error("Detector not loaded. Call load() first.")
            return []
        
        image = frame.image
        
        # Run YOLO inference
        results = self._model(
            image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Collect valid detections and their crops
        detections_data: List[tuple] = []  # (bbox, class_id, confidence, class_name)
        crops: List[np.ndarray] = []
        filtered_count = 0
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(xyxy)):
                class_id = int(cls_ids[i])
                
                # Apply class filter
                if not config.detection.is_class_allowed(class_id):
                    filtered_count += 1
                    continue
                
                bbox = xyxy[i].astype(np.float32)
                confidence = float(confs[i])
                class_name = self._class_names.get(class_id, f"class_{class_id}")
                
                # Extract crop for embedding
                crop = self._extract_crop(image, bbox)
                
                detections_data.append((bbox, class_id, confidence, class_name))
                crops.append(crop)
        
        # Batch encode all crops
        if crops and self._encoder is not None:
            embeddings = self._encoder.encode(crops)
        else:
            embeddings = np.zeros((len(crops), self._encoder.embedding_dim if self._encoder else 64), dtype=np.float32)
        
        # Build Detection objects
        detections: List[Detection] = []
        for i, (bbox, class_id, confidence, class_name) in enumerate(detections_data):
            detection = Detection(
                bbox=bbox,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                embedding=embeddings[i] if i < len(embeddings) else None
            )
            detections.append(detection)
        
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} static objects")
        
        return detections
    
    def _extract_crop(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract object crop from image."""
        h, w = image.shape[:2]
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(w, bbox[2]))
        y2 = int(min(h, bbox[3]))
        
        if x2 - x1 < MIN_CROP_SIZE or y2 - y1 < MIN_CROP_SIZE:
            return np.zeros((MIN_CROP_SIZE, MIN_CROP_SIZE, 3), dtype=np.uint8)
        
        return image[y1:y2, x1:x2].copy()
    
    @property
    def class_names(self) -> dict[int, str]:
        """Get mapping of class ID to class name."""
        return self._class_names.copy()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    @property
    def encoder_type(self) -> str:
        """Get the type of encoder being used."""
        if self._encoder is None:
            return "none"
        return type(self._encoder).__name__
