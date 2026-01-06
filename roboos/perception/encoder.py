"""
Encoder abstraction layer for semantic embeddings.

Provides pluggable encoder implementations:
    - VJEPAEncoder: V-JEPA 2 from HuggingFace (production)
    - HistogramEncoder: Color histogram fallback (fast, CPU)

Design Principles:
    1. Protocol-based interface for easy swapping
    2. Graceful fallback when V-JEPA unavailable
    3. Batch processing for efficiency
    4. GPU memory management

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, runtime_checkable, Final
from dataclasses import dataclass
from enum import Enum
import cv2
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, V-JEPA encoder disabled")


# =============================================================================
# Constants
# =============================================================================

# Default embedding dimension (V-JEPA ViT-S outputs 384-dim)
DEFAULT_EMBEDDING_DIM: Final[int] = 384

# Histogram encoder settings
HISTOGRAM_BINS: Final[int] = 32
HISTOGRAM_EMBEDDING_DIM: Final[int] = 64

# V-JEPA model variants (from HuggingFace)
# Reference: https://huggingface.co/facebook
VJEPA_MODEL_LARGE: Final[str] = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_MODEL_GIANT: Final[str] = "facebook/vjepa2-vitg-fpc64-384-ssv2"
VJEPA_MODEL_DIVING: Final[str] = "facebook/vjepa2-vitg-fpc32-384-diving48"

# Default model (large is most practical for real-time)
VJEPA_MODEL_DEFAULT: Final[str] = VJEPA_MODEL_LARGE


# =============================================================================
# Encoder Types
# =============================================================================

class EncoderType(str, Enum):
    """Available encoder implementations."""
    VJEPA = "vjepa"
    HISTOGRAM = "histogram"
    AUTO = "auto"  # Try V-JEPA, fallback to histogram


# =============================================================================
# Encoder Protocol
# =============================================================================

@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for image encoders.
    
    Any class implementing this protocol can be used for embedding extraction.
    """
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of output embeddings."""
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Whether the encoder is ready to use."""
        ...
    
    def load(self) -> bool:
        """Load model weights. Returns True if successful."""
        ...
    
    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: List of BGR images (any size)
            
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        ...
    
    def encode_single(self, image: np.ndarray) -> np.ndarray:
        """
        Encode a single image.
        
        Args:
            image: BGR image (any size)
            
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        ...


# =============================================================================
# Histogram Encoder (Fallback)
# =============================================================================

class HistogramEncoder:
    """
    Color histogram-based encoder.
    
    Simple, fast, and works on CPU. Used as fallback when V-JEPA unavailable.
    Not as good for re-identification but reliable.
    """
    
    __slots__ = ('_embedding_dim', '_target_size', '_loaded')
    
    def __init__(self, embedding_dim: int = HISTOGRAM_EMBEDDING_DIM):
        self._embedding_dim = embedding_dim
        self._target_size = (64, 64)
        self._loaded = True  # No weights to load
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> bool:
        """No-op for histogram encoder."""
        logger.info("HistogramEncoder ready (no weights to load)")
        return True
    
    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        """Encode batch of images to histogram embeddings."""
        if not images:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        
        embeddings = np.zeros((len(images), self._embedding_dim), dtype=np.float32)
        
        for i, img in enumerate(images):
            embeddings[i] = self.encode_single(img)
        
        return embeddings
    
    def encode_single(self, image: np.ndarray) -> np.ndarray:
        """Encode single image to histogram embedding."""
        if image is None or image.size == 0:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        
        # Resize for consistency
        resized = cv2.resize(image, self._target_size)
        
        # Convert to HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Compute histograms
        h_hist = cv2.calcHist([hsv], [0], None, [HISTOGRAM_BINS], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [HISTOGRAM_BINS], [0, 256])
        
        # Normalize
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        
        # Combine
        embedding = np.concatenate([h_hist.flatten(), s_hist.flatten()])
        
        # Pad or truncate
        if len(embedding) < self._embedding_dim:
            embedding = np.pad(embedding, (0, self._embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self._embedding_dim]
        
        return embedding.astype(np.float32)


# =============================================================================
# V-JEPA Encoder
# =============================================================================

class VJEPAEncoder:
    """
    V-JEPA 2 encoder using HuggingFace Transformers.
    
    Produces semantic embeddings that understand object appearance,
    motion patterns, and temporal context.
    
    Model Variants:
        - vits (small): 86MB, ~10ms/frame, good quality
        - vitl (large): 300MB, ~25ms/frame, better quality  
        - vitg (giant): 1.8GB, ~50ms/frame, best quality
    """
    
    __slots__ = (
        '_model_name', '_device', '_embedding_dim',
        '_model', '_processor', '_loaded', '_target_size'
    )
    
    def __init__(
        self,
        model_name: str = VJEPA_MODEL_DEFAULT,
        device: str = "cuda"
    ):
        self._model_name = model_name
        self._device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._target_size = (224, 224)
        
        self._model = None
        self._processor = None
        self._loaded = False
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> bool:
        """
        Load V-JEPA model from HuggingFace.
        
        Downloads weights on first use (~300MB for large model).
        Requires: pip install -U git+https://github.com/huggingface/transformers
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot load V-JEPA")
            return False
        
        try:
            from transformers import AutoVideoProcessor, AutoModel
            
            logger.info(f"Loading V-JEPA model: {self._model_name}")
            logger.info(f"Device: {self._device}")
            
            # Load processor and model using V-JEPA specific classes
            self._processor = AutoVideoProcessor.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16  # Use FP16 for 2x speed
            )
            self._model.to(self._device)
            self._model.eval()
            
            # Get actual embedding dimension from model config
            self._embedding_dim = self._model.config.hidden_size
            
            logger.info(
                f"V-JEPA loaded: embedding_dim={self._embedding_dim}, "
                f"device={self._device}, dtype=float16"
            )
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load V-JEPA: {e}")
            logger.warning("Falling back to histogram encoder")
            return False
    
    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Encode batch of images to V-JEPA embeddings.
        
        V-JEPA is designed for video, so we repeat each image
        to create a pseudo-video for encoding.
        
        Args:
            images: List of BGR images (any size)
            
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        if not self._loaded or not images:
            return np.zeros((len(images) if images else 0, self._embedding_dim), dtype=np.float32)
        
        embeddings_list = []
        
        try:
            for img in images:
                # Preprocess single image
                embedding = self._encode_single_image(img)
                embeddings_list.append(embedding)
            
            return np.stack(embeddings_list, axis=0)
            
        except Exception as e:
            logger.error(f"V-JEPA encode failed: {e}")
            return np.zeros((len(images), self._embedding_dim), dtype=np.float32)
    
    def _encode_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode a single image using V-JEPA.
        
        V-JEPA returns spatiotemporal features of shape (B, T*H*W, D).
        We pool across the spatial-temporal dimension to get (B, D).
        """
        if image is None or image.size == 0:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, self._target_size)
        
        # Process image - V-JEPA expects video, so we repeat the image
        # Using 4 frames (reduced from 16) for faster inference
        inputs = self._processor(rgb_image, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(self._device, dtype=torch.float16)
        
        # Repeat image 8 times (balanced: 16 = quality, 4 = fast, 8 = balanced)
        pixel_values = pixel_values.repeat(1, 8, 1, 1, 1)
        
        # Get vision features using the official API with autocast
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self._model.get_vision_features(pixel_values)
        
        # Global average pooling across the patch dimension
        # (B, num_patches, D) -> (B, D)
        embedding = features.mean(dim=1)
        
        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.float().cpu().numpy().flatten().astype(np.float32)
    
    def encode_single(self, image: np.ndarray) -> np.ndarray:
        """Encode single image."""
        if not self._loaded:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        return self._encode_single_image(image)


# =============================================================================
# Encoder Factory
# =============================================================================

@dataclass
class EncoderConfig:
    """Configuration for encoder selection."""
    encoder_type: EncoderType = EncoderType.AUTO
    vjepa_model: str = VJEPA_MODEL_DEFAULT
    device: str = "cuda"
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


def create_encoder(config: Optional[EncoderConfig] = None) -> Encoder:
    """
    Create an encoder based on configuration.
    
    Args:
        config: Encoder configuration. If None, uses AUTO mode.
        
    Returns:
        Loaded encoder instance
        
    Example:
        encoder = create_encoder()
        embeddings = encoder.encode([img1, img2])
    """
    if config is None:
        config = EncoderConfig()
    
    if config.encoder_type == EncoderType.HISTOGRAM:
        encoder = HistogramEncoder(embedding_dim=config.embedding_dim)
        encoder.load()
        return encoder
    
    if config.encoder_type == EncoderType.VJEPA:
        encoder = VJEPAEncoder(model_name=config.vjepa_model, device=config.device)
        if encoder.load():
            return encoder
        raise RuntimeError("Failed to load V-JEPA encoder")
    
    # AUTO mode: try V-JEPA, fallback to histogram
    if TORCH_AVAILABLE:
        try:
            encoder = VJEPAEncoder(model_name=config.vjepa_model, device=config.device)
            if encoder.load():
                return encoder
        except Exception as e:
            logger.warning(f"V-JEPA failed, using histogram: {e}")
    
    # Fallback
    encoder = HistogramEncoder(embedding_dim=config.embedding_dim)
    encoder.load()
    return encoder
