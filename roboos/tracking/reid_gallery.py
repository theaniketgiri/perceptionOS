"""
Re-Identification Gallery for persistent object identity.

This module stores embeddings of lost objects and enables re-identification
when they reappear after prolonged absence.

Architecture:
    - Real-time tracking uses histogram embeddings (fast)
    - Gallery stores V-JEPA embeddings (semantic, for re-ID)
    - When new detection appears, compare against gallery
    - Match found → Resurrect old ID instead of creating new

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List, Final
from dataclasses import dataclass, field
import time
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

# Gallery retention time (milliseconds)
GALLERY_RETENTION_MS: Final[int] = 60000  # Keep for 60 seconds

# Re-ID similarity threshold
REID_SIMILARITY_THRESHOLD: Final[float] = 0.7  # Must be > 70% similar

# Maximum gallery size
MAX_GALLERY_SIZE: Final[int] = 50


# =============================================================================
# Gallery Entry
# =============================================================================

@dataclass
class GalleryEntry:
    """
    Stored embedding for a lost object.
    
    Attributes:
        object_id: Original object ID (e.g., "obj_0001")
        class_id: COCO class ID
        class_name: Human-readable class name
        embedding: V-JEPA semantic embedding
        last_position: Last known position [x, y]
        lost_timestamp_ms: When the object was lost
    """
    object_id: str
    class_id: int
    class_name: str
    embedding: np.ndarray
    last_position: np.ndarray
    lost_timestamp_ms: int


# =============================================================================
# Re-ID Gallery
# =============================================================================

class ReIDGallery:
    """
    Gallery of lost object embeddings for re-identification.
    
    When an object is lost (removed from tracker), its embedding
    is stored here. When a new detection appears, we check if it
    matches any gallery entry before creating a new ID.
    
    Usage:
        gallery = ReIDGallery()
        
        # When object is lost
        gallery.add(object_id, class_id, class_name, embedding, position, timestamp)
        
        # When new detection appears
        match = gallery.find_match(new_embedding, class_id, position, timestamp)
        if match:
            # Resurrect old ID instead of creating new
            old_id = match.object_id
    """
    
    __slots__ = ('_entries', '_encoder')
    
    def __init__(self):
        self._entries: Dict[str, GalleryEntry] = {}
        self._encoder = None  # V-JEPA encoder (lazy loaded)
    
    def set_encoder(self, encoder) -> None:
        """Set the V-JEPA encoder for semantic embeddings."""
        self._encoder = encoder
        logger.info("ReIDGallery: V-JEPA encoder set for re-identification")
    
    def add(
        self,
        object_id: str,
        class_id: int,
        class_name: str,
        embedding: np.ndarray,
        last_position: np.ndarray,
        lost_timestamp_ms: int
    ) -> None:
        """
        Add a lost object to the gallery.
        
        Args:
            object_id: Original object ID
            class_id: COCO class ID
            class_name: Human-readable class name
            embedding: Semantic embedding (V-JEPA preferred)
            last_position: Last known position [x, y]
            lost_timestamp_ms: When the object was lost
        """
        # Enforce gallery size limit
        if len(self._entries) >= MAX_GALLERY_SIZE:
            self._evict_oldest()
        
        entry = GalleryEntry(
            object_id=object_id,
            class_id=class_id,
            class_name=class_name,
            embedding=embedding.copy() if embedding is not None else None,
            last_position=last_position.copy() if last_position is not None else None,
            lost_timestamp_ms=lost_timestamp_ms
        )
        
        self._entries[object_id] = entry
        logger.info(
            f"ReIDGallery: Added {object_id} ({class_name}) - "
            f"gallery size: {len(self._entries)}"
        )
    
    def find_match(
        self,
        embedding: np.ndarray,
        class_id: int,
        position: np.ndarray,
        current_timestamp_ms: int
    ) -> Optional[GalleryEntry]:
        """
        Find a matching entry in the gallery.
        
        Args:
            embedding: Embedding of new detection
            class_id: Class ID of new detection
            position: Position of new detection
            current_timestamp_ms: Current timestamp
            
        Returns:
            Matching GalleryEntry or None
        """
        if embedding is None or len(self._entries) == 0:
            return None
        
        # Clean expired entries first
        self._cleanup(current_timestamp_ms)
        
        best_match: Optional[GalleryEntry] = None
        best_similarity = REID_SIMILARITY_THRESHOLD
        
        for entry in self._entries.values():
            # Must be same class
            if entry.class_id != class_id:
                continue
            
            # Must have valid embedding
            if entry.embedding is None:
                continue
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, entry.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            logger.info(
                f"ReIDGallery: Match found! {best_match.object_id} "
                f"(similarity: {best_similarity:.2f})"
            )
        
        return best_match
    
    def remove(self, object_id: str) -> None:
        """Remove an entry from the gallery (e.g., after successful re-ID)."""
        if object_id in self._entries:
            del self._entries[object_id]
            logger.debug(f"ReIDGallery: Removed {object_id}")
    
    def _cleanup(self, current_timestamp_ms: int) -> None:
        """Remove expired entries."""
        expired = [
            obj_id for obj_id, entry in self._entries.items()
            if current_timestamp_ms - entry.lost_timestamp_ms > GALLERY_RETENTION_MS
        ]
        
        for obj_id in expired:
            del self._entries[obj_id]
            logger.debug(f"ReIDGallery: Expired {obj_id}")
    
    def _evict_oldest(self) -> None:
        """Remove the oldest entry to make room."""
        if not self._entries:
            return
        
        oldest_id = min(
            self._entries.keys(),
            key=lambda x: self._entries[x].lost_timestamp_ms
        )
        del self._entries[oldest_id]
        logger.debug(f"ReIDGallery: Evicted {oldest_id}")
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @property
    def size(self) -> int:
        """Number of entries in the gallery."""
        return len(self._entries)
    
    def get_all_ids(self) -> List[str]:
        """Get all object IDs in the gallery."""
        return list(self._entries.keys())
