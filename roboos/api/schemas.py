"""
Pydantic schemas for API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ObjectStateEnum(str, Enum):
    """State of a tracked object"""
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    UNCERTAIN = "uncertain"
    LOST = "lost"


class TrackedObjectSchema(BaseModel):
    """Schema for a single tracked object"""
    id: str = Field(..., description="Unique object identifier")
    object_class: str = Field(..., alias="class", description="Object class (e.g., person, car)")
    state: ObjectStateEnum = Field(..., description="Current tracking state")
    position: List[float] = Field(..., description="[x, y] position in pixels")
    velocity: List[float] = Field(..., description="[vx, vy] velocity in pixels/frame")
    last_seen_ms: int = Field(..., description="Milliseconds since last observation")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    identity_confidence: float = Field(..., ge=0, le=1, description="Identity confidence after occlusion")
    
    class Config:
        populate_by_name = True


class WorldStateResponse(BaseModel):
    """Full world state response"""
    timestamp_ms: int = Field(..., description="Timestamp of this state")
    frame_id: int = Field(..., description="Frame number")
    objects: List[TrackedObjectSchema] = Field(..., description="All tracked objects")


class WorldStateSummary(BaseModel):
    """Summary of world state"""
    total_objects: int = Field(..., description="Total number of tracked objects")
    tracked: int = Field(0, description="Objects currently being tracked")
    occluded: int = Field(0, description="Objects currently occluded")
    uncertain: int = Field(0, description="Objects with uncertain state")
    lost: int = Field(0, description="Objects that have been lost")


class HealthResponse(BaseModel):
    """System health response"""
    status: str = Field(..., description="System status (ok/error)")
    fps: float = Field(..., description="Current frames per second")
    frame_count: int = Field(..., description="Total frames processed")
    gpu: str = Field(..., description="GPU device name")
    objects_tracked: int = Field(..., description="Number of objects being tracked")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional detail")
