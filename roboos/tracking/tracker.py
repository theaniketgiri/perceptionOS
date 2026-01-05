"""
Multi-object tracker with identity persistence through occlusions.

Design Principles:
    1. Identity is sacred: Never silently reassign IDs
    2. Uncertainty is explicit: Low confidence = state change, not ID change  
    3. Fail auditably: Every decision is logged and traceable
    4. Predictable behavior: Deterministic given same inputs

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Tuple, Final
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from scipy.optimize import linear_sum_assignment
from loguru import logger

from roboos.config import config
from roboos.perception.detector import Detection
from roboos.tracking.kalman import KalmanFilter


# =============================================================================
# Constants - All magic numbers live here
# =============================================================================

# Cost matrix weights for Hungarian matching
COST_WEIGHT_POSITION: Final[float] = 0.5   # Increased from 0.4
COST_WEIGHT_EMBEDDING: Final[float] = 0.35  # Adjusted
COST_WEIGHT_CLASS: Final[float] = 0.15      # Reduced from 0.2

# Thresholds for matching decisions
MATCH_COST_THRESHOLD: Final[float] = 0.8  # Below this = valid match
IDENTITY_CONFIDENCE_THRESHOLD: Final[float] = 0.5  # Below this = uncertain match
EMBEDDING_SIMILARITY_MIN: Final[float] = 0.3  # Below this = reject match

# Position normalization factor (pixels)
POSITION_NORM_FACTOR: Final[float] = 500.0

# Embedding update rate (exponential moving average)
EMBEDDING_UPDATE_ALPHA: Final[float] = 0.3

# Confidence decay rates per frame - SLOWED DOWN
CONFIDENCE_DECAY_TRACKED: Final[float] = 0.02   # Was 0.05
CONFIDENCE_DECAY_OCCLUDED: Final[float] = 0.015  # Was 0.03
CONFIDENCE_DECAY_UNCERTAIN: Final[float] = 0.005 # Was 0.01

# Minimum confidence floors
CONFIDENCE_FLOOR_TRACKED: Final[float] = 0.6
CONFIDENCE_FLOOR_OCCLUDED: Final[float] = 0.3
CONFIDENCE_FLOOR_UNCERTAIN: Final[float] = 0.1


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ObjectState(str, Enum):
    """
    State machine for tracked objects.
    
    Transitions:
        TRACKED -> OCCLUDED: Object not detected but predicted
        OCCLUDED -> UNCERTAIN: Confidence too low for reliable prediction
        UNCERTAIN -> LOST: Object has been uncertain too long
        Any -> TRACKED: Object re-detected with high confidence match
    
    CRITICAL: State changes do NOT change object ID. Ever.
    """
    TRACKED = "tracked"
    OCCLUDED = "occluded"  
    UNCERTAIN = "uncertain"
    LOST = "lost"


class MatchDecision(Enum):
    """Result of attempting to match a detection to a track."""
    ACCEPT = auto()           # High confidence match
    ACCEPT_UNCERTAIN = auto() # Match accepted but identity uncertain
    REJECT = auto()           # No valid match


@dataclass(slots=True)
class TrackedObject:
    """
    A persistent tracked object with guaranteed identity.
    
    Invariants:
        - `id` is immutable after creation
        - `identity_confidence` reflects certainty about ID, not detection quality
        - `state` reflects visibility, not identity certainty
    """
    id: str
    class_id: int
    class_name: str
    
    kalman: KalmanFilter = field(default_factory=KalmanFilter)
    embedding: Optional[np.ndarray] = None
    
    state: ObjectState = ObjectState.TRACKED
    confidence: float = 1.0
    identity_confidence: float = 1.0
    
    last_seen_ms: int = 0
    created_at_ms: int = 0
    
    hits: int = 0
    misses: int = 0
    
    # Audit trail
    _match_history: List[Tuple[int, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate invariants on creation."""
        if not self.id:
            raise ValueError("TrackedObject must have a non-empty ID")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y]."""
        return self.kalman.state.position
    
    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vx, vy]."""
        return self.kalman.state.velocity
    
    @property
    def age_ms(self) -> int:
        """Time since object was first tracked."""
        return int(time.time() * 1000) - self.created_at_ms
    
    @property
    def time_since_seen_ms(self) -> int:
        """Time since object was last directly observed."""
        return int(time.time() * 1000) - self.last_seen_ms
    
    @property
    def is_reliable(self) -> bool:
        """True if identity and detection are both confident."""
        return (
            self.identity_confidence >= IDENTITY_CONFIDENCE_THRESHOLD
            and self.state == ObjectState.TRACKED
        )
    
    def record_match(self, frame_id: int, match_cost: float) -> None:
        """Record a match for audit purposes."""
        self._match_history.append((frame_id, match_cost))
        # Keep only last 100 matches to bound memory
        if len(self._match_history) > 100:
            self._match_history = self._match_history[-100:]
    
    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "id": self.id,
            "class": self.class_name,
            "state": self.state.value,
            "position": [round(p, 2) for p in self.position.tolist()],
            "velocity": [round(v, 3) for v in self.velocity.tolist()],
            "last_seen_ms": self.time_since_seen_ms,
            "confidence": round(self.confidence, 3),
            "identity_confidence": round(self.identity_confidence, 3),
        }


# =============================================================================
# Tracker Implementation
# =============================================================================

class Tracker:
    """
    Multi-object tracker with strict identity preservation.
    
    Core Guarantees:
        1. Object IDs are never reused or reassigned
        2. Low identity_confidence triggers UNCERTAIN state, not ID change
        3. All matching decisions are auditable via logs
        4. Deterministic behavior given identical inputs
    
    Architecture:
        Detection -> Cost Matrix -> Hungarian Match -> State Update
                                         |
                                         v
                              Match Validation Gate
                              (rejects low-confidence matches)
    """
    
    __slots__ = (
        '_objects', '_next_id', '_frame_id', '_max_age', '_min_hits',
        '_iou_threshold', '_embedding_threshold', '_occlusion_threshold',
        '_uncertain_threshold', '_lost_threshold'
    )
    
    def __init__(self) -> None:
        self._objects: Dict[str, TrackedObject] = {}
        self._next_id: int = 1
        self._frame_id: int = 0
        
        # Load from config
        self._max_age: int = config.tracker.max_age
        self._min_hits: int = config.tracker.min_hits
        self._iou_threshold: float = config.tracker.iou_threshold
        self._embedding_threshold: float = config.tracker.embedding_threshold
        
        self._occlusion_threshold: int = config.tracker.occlusion_threshold_ms
        self._uncertain_threshold: int = config.tracker.uncertain_threshold_ms
        self._lost_threshold: int = config.tracker.lost_threshold_ms
    
    def _generate_id(self) -> str:
        """
        Generate a unique, never-reused object ID.
        
        Format: obj_XXXX where XXXX is zero-padded sequential number.
        IDs are strictly monotonic and never recycled.
        """
        obj_id = f"obj_{self._next_id:04d}"
        self._next_id += 1
        return obj_id
    
    def update(
        self,
        detections: List[Detection],
        timestamp_ms: int,
        frame_id: int
    ) -> List[TrackedObject]:
        """
        Process new detections and update all track states.
        
        Args:
            detections: Objects detected in current frame
            timestamp_ms: Timestamp of current frame
            frame_id: Sequential frame identifier
            
        Returns:
            All tracked objects including occluded/uncertain ones
            
        Note:
            This method is the single entry point for all state updates.
            No external code should modify TrackedObject state directly.
        """
        self._frame_id = frame_id
        
        # Phase 1: Predict all existing tracks forward
        self._predict_all_tracks()
        
        # Phase 2: Match detections to tracks with validation
        matched, unmatched_dets, unmatched_tracks = self._match_detections_validated(
            detections, timestamp_ms
        )
        
        # Phase 3: Update matched tracks
        for det_idx, obj_id, match_cost in matched:
            self._update_matched_track(
                obj_id, detections[det_idx], timestamp_ms, match_cost
            )
        
        # Phase 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], timestamp_ms)
        
        # Phase 5: Handle unmatched tracks (occlusions)
        for obj_id in unmatched_tracks:
            self._handle_unmatched_track(obj_id, timestamp_ms)
        
        # Phase 6: Garbage collect lost tracks
        self._remove_lost_tracks()
        
        return list(self._objects.values())
    
    def _predict_all_tracks(self) -> None:
        """Advance all Kalman filters by one timestep."""
        for obj in self._objects.values():
            obj.kalman.predict()
    
    def _match_detections_validated(
        self,
        detections: List[Detection],
        timestamp_ms: int
    ) -> Tuple[List[Tuple[int, str, float]], List[int], List[str]]:
        """
        Match detections to tracks with identity validation.
        
        Unlike naive Hungarian matching, this method:
        1. Computes optimal assignment
        2. Validates each match against identity thresholds
        3. Rejects matches that would compromise identity integrity
        
        Returns:
            matched: List of (det_idx, obj_id, match_cost) for valid matches
            unmatched_dets: Detection indices without valid matches
            unmatched_tracks: Object IDs without valid matches
        """
        if not detections:
            return [], [], list(self._objects.keys())
        
        if not self._objects:
            return [], list(range(len(detections))), []
        
        # Build cost matrix
        obj_ids = list(self._objects.keys())
        cost_matrix = self._build_cost_matrix(detections, obj_ids)
        
        # Solve assignment problem
        det_indices, obj_indices = linear_sum_assignment(cost_matrix)
        
        # Validate each match
        matched: List[Tuple[int, str, float]] = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(obj_ids)
        
        for det_idx, obj_idx in zip(det_indices, obj_indices):
            cost = cost_matrix[det_idx, obj_idx]
            obj_id = obj_ids[obj_idx]
            obj = self._objects[obj_id]
            det = detections[det_idx]
            
            decision = self._validate_match(det, obj, cost)
            
            if decision == MatchDecision.ACCEPT:
                matched.append((det_idx, obj_id, cost))
                unmatched_dets.discard(det_idx)
                unmatched_tracks.discard(obj_id)
                logger.debug(
                    f"Frame {self._frame_id}: Matched det[{det_idx}] -> {obj_id} "
                    f"(cost={cost:.3f})"
                )
                
            elif decision == MatchDecision.ACCEPT_UNCERTAIN:
                # Accept match BUT mark identity as uncertain
                # CRITICAL: We do NOT create a new ID. We keep the existing ID
                # but flag that we're not sure it's the same object.
                matched.append((det_idx, obj_id, cost))
                unmatched_dets.discard(det_idx)
                unmatched_tracks.discard(obj_id)
                logger.warning(
                    f"Frame {self._frame_id}: Uncertain match det[{det_idx}] -> {obj_id} "
                    f"(cost={cost:.3f}, identity_confidence will degrade)"
                )
                
            else:  # REJECT
                logger.debug(
                    f"Frame {self._frame_id}: Rejected match det[{det_idx}] -> {obj_id} "
                    f"(cost={cost:.3f} exceeds threshold)"
                )
        
        return matched, list(unmatched_dets), list(unmatched_tracks)
    
    def _build_cost_matrix(
        self,
        detections: List[Detection],
        obj_ids: List[str]
    ) -> np.ndarray:
        """
        Build cost matrix for Hungarian algorithm.
        
        Cost components:
            - Position distance (normalized by POSITION_NORM_FACTOR)
            - Embedding dissimilarity (1 - cosine_similarity)
            - Class mismatch penalty (0 or 1)
        """
        n_dets = len(detections)
        n_objs = len(obj_ids)
        cost_matrix = np.full((n_dets, n_objs), fill_value=999.0, dtype=np.float64)
        
        for i, det in enumerate(detections):
            for j, obj_id in enumerate(obj_ids):
                obj = self._objects[obj_id]
                
                # Position cost
                pos_dist = float(np.linalg.norm(det.center - obj.position))
                pos_cost = min(pos_dist / POSITION_NORM_FACTOR, 2.0)
                
                # Embedding cost
                emb_cost = self._compute_embedding_cost(det.embedding, obj.embedding)
                
                # Class cost
                class_cost = 0.0 if det.class_id == obj.class_id else 1.0
                
                # Weighted sum
                cost_matrix[i, j] = (
                    COST_WEIGHT_POSITION * pos_cost +
                    COST_WEIGHT_EMBEDDING * emb_cost +
                    COST_WEIGHT_CLASS * class_cost
                )
        
        return cost_matrix
    
    def _compute_embedding_cost(
        self,
        det_emb: Optional[np.ndarray],
        obj_emb: Optional[np.ndarray]
    ) -> float:
        """
        Compute embedding dissimilarity cost.
        
        Returns:
            Cost in [0, 1] where 0 = identical, 1 = completely different
        """
        if det_emb is None or obj_emb is None:
            return 0.5  # Neutral cost when embeddings unavailable
        
        # Cosine similarity
        det_norm = np.linalg.norm(det_emb)
        obj_norm = np.linalg.norm(obj_emb)
        
        if det_norm < 1e-8 or obj_norm < 1e-8:
            return 0.5
        
        similarity = float(np.dot(det_emb, obj_emb) / (det_norm * obj_norm))
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return (1.0 - similarity) / 2.0  # Map [-1, 1] -> [1, 0] -> [0, 1]
    
    def _validate_match(
        self,
        detection: Detection,
        obj: TrackedObject,
        cost: float
    ) -> MatchDecision:
        """
        Validate whether a match should be accepted.
        
        Decision Logic:
            - cost > MATCH_COST_THRESHOLD -> REJECT
            - embedding_similarity < EMBEDDING_SIMILARITY_MIN -> REJECT  
            - obj.identity_confidence < threshold -> ACCEPT_UNCERTAIN
            - otherwise -> ACCEPT
            
        CRITICAL: This is the gate that prevents bad ID reassignments.
        """
        # Hard reject if cost too high
        if cost > MATCH_COST_THRESHOLD:
            return MatchDecision.REJECT
        
        # Check embedding similarity if available
        if detection.embedding is not None and obj.embedding is not None:
            similarity = self._compute_cosine_similarity(
                detection.embedding, obj.embedding
            )
            if similarity < EMBEDDING_SIMILARITY_MIN:
                return MatchDecision.REJECT
        
        # If object was uncertain and this match is marginal, flag it
        if obj.state == ObjectState.UNCERTAIN:
            return MatchDecision.ACCEPT_UNCERTAIN
        
        # If match cost is in the marginal zone, accept but uncertain
        if cost > MATCH_COST_THRESHOLD * 0.7:
            return MatchDecision.ACCEPT_UNCERTAIN
        
        return MatchDecision.ACCEPT
    
    def _compute_cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _update_matched_track(
        self,
        obj_id: str,
        detection: Detection,
        timestamp_ms: int,
        match_cost: float
    ) -> None:
        """
        Update a track that was successfully matched to a detection.
        
        Updates:
            - Kalman filter with measurement
            - Embedding via exponential moving average
            - State and confidence based on match quality
        """
        obj = self._objects[obj_id]
        
        # Update Kalman filter
        center = detection.center
        obj.kalman.update(float(center[0]), float(center[1]))
        
        # Update embedding with EMA
        if detection.embedding is not None:
            if obj.embedding is None:
                obj.embedding = detection.embedding.copy()
            else:
                obj.embedding = (
                    (1 - EMBEDDING_UPDATE_ALPHA) * obj.embedding +
                    EMBEDDING_UPDATE_ALPHA * detection.embedding
                )
        
        # Determine if this was an uncertain match
        is_uncertain_match = match_cost > MATCH_COST_THRESHOLD * 0.7
        
        # Update state
        obj.state = ObjectState.TRACKED
        obj.confidence = detection.confidence
        obj.last_seen_ms = timestamp_ms
        obj.hits += 1
        obj.misses = 0
        
        # CRITICAL: Update identity confidence based on match quality
        if is_uncertain_match:
            # Degrade identity confidence for marginal matches
            obj.identity_confidence = max(
                0.3,
                obj.identity_confidence - 0.1
            )
            logger.debug(
                f"{obj_id}: Uncertain match, identity_confidence -> "
                f"{obj.identity_confidence:.2f}"
            )
        else:
            # Restore identity confidence for good matches
            obj.identity_confidence = min(
                1.0,
                obj.identity_confidence + 0.05
            )
        
        obj.record_match(self._frame_id, match_cost)
    
    def _create_track(self, detection: Detection, timestamp_ms: int) -> None:
        """
        Create a new track from an unmatched detection.
        
        New tracks start with:
            - identity_confidence = 1.0 (we're certain of new identity)
            - state = TRACKED
            - hits = 1
        """
        obj_id = self._generate_id()
        
        embedding = None
        if detection.embedding is not None:
            embedding = detection.embedding.copy()
        
        obj = TrackedObject(
            id=obj_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            embedding=embedding,
            state=ObjectState.TRACKED,
            confidence=detection.confidence,
            identity_confidence=1.0,  # New track = certain identity
            last_seen_ms=timestamp_ms,
            created_at_ms=timestamp_ms,
            hits=1,
            misses=0,
        )
        
        center = detection.center
        obj.kalman.initialize(float(center[0]), float(center[1]))
        
        self._objects[obj_id] = obj
        
        logger.info(
            f"Frame {self._frame_id}: Created track {obj_id} "
            f"({detection.class_name} @ [{center[0]:.0f}, {center[1]:.0f}])"
        )
    
    def _handle_unmatched_track(self, obj_id: str, timestamp_ms: int) -> None:
        """
        Handle a track that wasn't matched to any detection.
        
        State Transitions:
            TRACKED -> OCCLUDED (if recently seen)
            OCCLUDED -> UNCERTAIN (if identity_confidence low)
            UNCERTAIN -> LOST (if too long)
        """
        obj = self._objects[obj_id]
        obj.misses += 1
        
        time_since_seen = timestamp_ms - obj.last_seen_ms
        
        if time_since_seen < self._occlusion_threshold:
            # Recently seen - likely temporary occlusion
            obj.state = ObjectState.OCCLUDED
            obj.confidence = max(
                CONFIDENCE_FLOOR_TRACKED,
                obj.confidence - CONFIDENCE_DECAY_TRACKED
            )
            
        elif time_since_seen < self._uncertain_threshold:
            # Getting concerning - identity less certain
            obj.state = ObjectState.OCCLUDED
            obj.confidence = max(
                CONFIDENCE_FLOOR_OCCLUDED,
                obj.confidence - CONFIDENCE_DECAY_OCCLUDED
            )
            obj.identity_confidence = max(
                IDENTITY_CONFIDENCE_THRESHOLD,
                obj.identity_confidence - 0.05
            )
            
        elif time_since_seen < self._lost_threshold:
            # CRITICAL: Low identity confidence -> switch to UNCERTAIN state
            # We do NOT reassign or create new ID. We mark uncertainty.
            prev_state = obj.state
            obj.state = ObjectState.UNCERTAIN
            obj.confidence = max(
                CONFIDENCE_FLOOR_UNCERTAIN,
                obj.confidence - CONFIDENCE_DECAY_UNCERTAIN
            )
            obj.identity_confidence = max(
                0.1,
                obj.identity_confidence - 0.1
            )
            
            # Only log on state TRANSITION (not every frame)
            if prev_state != ObjectState.UNCERTAIN:
                logger.warning(
                    f"{obj_id}: Marked UNCERTAIN "
                    f"(identity_confidence={obj.identity_confidence:.2f}, "
                    f"not seen for {time_since_seen}ms)"
                )
            
        else:
            # Lost - will be garbage collected
            obj.state = ObjectState.LOST
            obj.confidence = 0.0
            logger.info(f"{obj_id}: Marked LOST after {time_since_seen}ms")
    
    def _remove_lost_tracks(self) -> None:
        """Remove tracks that have been lost."""
        lost_ids = [
            obj_id for obj_id, obj in self._objects.items()
            if obj.state == ObjectState.LOST or obj.misses > self._max_age
        ]
        
        for obj_id in lost_ids:
            logger.info(f"Frame {self._frame_id}: Removing {obj_id}")
            del self._objects[obj_id]
    
    # =========================================================================
    # Public Query Interface
    # =========================================================================
    
    def get_object(self, obj_id: str) -> Optional[TrackedObject]:
        """Get a specific object by ID, or None if not found."""
        return self._objects.get(obj_id)
    
    @property
    def objects(self) -> List[TrackedObject]:
        """All tracked objects including uncertain/occluded."""
        return list(self._objects.values())
    
    @property
    def active_objects(self) -> List[TrackedObject]:
        """Only objects in TRACKED or OCCLUDED state."""
        return [
            obj for obj in self._objects.values()
            if obj.state in (ObjectState.TRACKED, ObjectState.OCCLUDED)
        ]
    
    @property
    def reliable_objects(self) -> List[TrackedObject]:
        """Only objects with high identity confidence."""
        return [obj for obj in self._objects.values() if obj.is_reliable]
    
    def get_summary(self) -> dict:
        """Get summary statistics for monitoring."""
        states: Dict[str, int] = {}
        low_identity_count = 0
        
        for obj in self._objects.values():
            states[obj.state.value] = states.get(obj.state.value, 0) + 1
            if obj.identity_confidence < IDENTITY_CONFIDENCE_THRESHOLD:
                low_identity_count += 1
        
        return {
            "total_objects": len(self._objects),
            "tracked": states.get("tracked", 0),
            "occluded": states.get("occluded", 0),
            "uncertain": states.get("uncertain", 0),
            "lost": states.get("lost", 0),
            "low_identity_confidence": low_identity_count,
        }
