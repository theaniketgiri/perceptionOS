"""
Object detector using YOLO for bounding box detection.

Responsibilities:
    1. Run YOLO inference on camera frames
    2. Filter detections by class (only track dynamic objects)
    3. Extract embeddings for re-identification

Design Notes:
    - Detection filtering happens at this layer, not in tracker
    - Embeddings use histogram for MVP (OSNet for production)
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


# =============================================================================
# Constants
# =============================================================================

EMBEDDING_CROP_SIZE: Final[int] = 64
HISTOGRAM_BINS: Final[int] = 32


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
    Object detector using YOLOv8 with class filtering.
    
    Features:
        - CUDA-accelerated inference
        - Configurable class filtering (only track dynamic objects)
        - Histogram-based embeddings for re-identification
    
    Usage:
        detector = Detector()
        detector.load()
        detections = detector.detect(frame)  # Already filtered
    """
    
    __slots__ = (
        'model_path', 'confidence_threshold', 'device',
        '_model', '_class_names', '_filter_enabled'
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
    
    def load(self) -> bool:
        """
        Load the YOLO model onto the specified device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading detector model: {self.model_path}")
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            self._class_names = self._model.names
            
            # Log configuration
            logger.info(f"Detector loaded on {self.device}, classes: {len(self._class_names)}")
            
            if self._filter_enabled:
                from roboos.config import get_allowed_class_names
                allowed = get_allowed_class_names()
                logger.info(f"Class filter enabled, tracking: {allowed}")
            else:
                logger.info("Class filter disabled, tracking all classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return False
    
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
        
        # Run inference
        results = self._model(
            frame.image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        detections: List[Detection] = []
        filtered_count = 0
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Extract arrays from YOLO output
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(xyxy)):
                class_id = int(cls_ids[i])
                
                # Apply class filter
                if not config.detection.is_class_allowed(class_id):
                    filtered_count += 1
                    continue
                
                bbox = xyxy[i]
                confidence = float(confs[i])
                class_name = self._class_names.get(class_id, f"class_{class_id}")
                
                # Extract embedding for re-identification
                embedding = self._extract_embedding(frame.image, bbox)
                
                detection = Detection(
                    bbox=bbox.astype(np.float32),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    embedding=embedding
                )
                detections.append(detection)
        
        # Log if we filtered anything (debug level)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} static objects")
        
        return detections
    
    def _extract_embedding(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """
        Extract a feature embedding from a detection region.
        
        For MVP: Uses color histogram (HSV)
        For Production: Would use OSNet, FastReID, or similar
        
        Args:
            image: Full BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Normalized feature vector
        """
        # Clip bbox to image bounds
        h, w = image.shape[:2]
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(w, bbox[2]))
        y2 = int(min(h, bbox[3]))
        
        # Handle degenerate boxes
        if x2 <= x1 or y2 <= y1:
            return np.zeros(config.detection.embedding_dim, dtype=np.float32)
        
        # Crop and resize
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (EMBEDDING_CROP_SIZE, EMBEDDING_CROP_SIZE))
        
        # Convert to HSV for color histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for H and S channels
        h_hist = cv2.calcHist([hsv], [0], None, [HISTOGRAM_BINS], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [HISTOGRAM_BINS], [0, 256])
        
        # Normalize histograms
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        
        # Combine into embedding
        embedding = np.concatenate([h_hist.flatten(), s_hist.flatten()])
        
        # Pad or truncate to target dimension
        target_dim = config.detection.embedding_dim
        if len(embedding) < target_dim:
            embedding = np.pad(embedding, (0, target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]
        
        return embedding.astype(np.float32)
    
    @property
    def class_names(self) -> dict[int, str]:
        """Get mapping of class ID to class name."""
        return self._class_names.copy()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
