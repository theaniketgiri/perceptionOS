"""
Object detector using YOLO for bounding box detection.
Extracts per-object embeddings for re-identification.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import torch
from ultralytics import YOLO
from loguru import logger

from roboos.config import config
from roboos.perception.camera import Frame


@dataclass
class Detection:
    """A single detected object"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    embedding: Optional[np.ndarray] = None  # Feature embedding for re-ID
    
    @property
    def center(self) -> np.ndarray:
        """Get center point of bounding box"""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height


class Detector:
    """
    Object detector using YOLOv8.
    Provides bounding boxes, class labels, and feature embeddings.
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        device: str = None
    ):
        self.model_path = model_path or config.model.detector
        self.confidence_threshold = confidence_threshold or config.model.detector_confidence
        self.device = device or config.model.device
        
        self._model: Optional[YOLO] = None
        self._class_names: dict = {}
        
    def load(self) -> bool:
        """Load the YOLO model"""
        try:
            logger.info(f"Loading detector model: {self.model_path}")
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            self._class_names = self._model.names
            logger.info(f"Detector loaded on {self.device}, classes: {len(self._class_names)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return False
    
    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input camera frame
            
        Returns:
            List of detections with bounding boxes and embeddings
        """
        if self._model is None:
            logger.error("Detector not loaded")
            return []
        
        # Run inference
        results = self._model(
            frame.image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Get bounding boxes, confidences, and class IDs
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            
            # Extract embeddings from model features if available
            # For now, we'll use the bounding box region as a simple embedding
            # In production, this would use a proper re-ID model
            for i in range(len(xyxy)):
                bbox = xyxy[i]
                class_id = cls_ids[i]
                confidence = float(confs[i])
                class_name = self._class_names.get(class_id, "unknown")
                
                # Extract embedding (placeholder - crop and resize)
                embedding = self._extract_embedding(frame.image, bbox)
                
                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    embedding=embedding
                )
                detections.append(detection)
        
        return detections
    
    def _extract_embedding(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        target_size: int = 64
    ) -> np.ndarray:
        """
        Extract a feature embedding from a detection region.
        
        For MVP, we use a simple histogram-based embedding.
        In production, this would use a proper re-ID model (e.g., OSNet).
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Size to resize crop to
            
        Returns:
            Feature embedding vector
        """
        import cv2
        
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(config.model.embedding_dim)
        
        # Crop and resize
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (target_size, target_size))
        
        # Convert to HSV for color histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for H and S channels
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        
        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        
        # Combine into embedding
        embedding = np.concatenate([h_hist, s_hist])
        
        # Pad or truncate to target dimension
        if len(embedding) < config.model.embedding_dim:
            embedding = np.pad(embedding, (0, config.model.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:config.model.embedding_dim]
        
        return embedding.astype(np.float32)
    
    @property
    def class_names(self) -> dict:
        return self._class_names
