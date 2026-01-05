"""
FastAPI server for world state API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from roboos.api.schemas import (
    WorldStateResponse,
    WorldStateSummary,
    TrackedObjectSchema,
    HealthResponse,
    ErrorResponse
)

# Create FastAPI app
app = FastAPI(
    title="Perception OS",
    description="Real-time world state perception API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# World state manager will be injected
_world_state_manager = None


def set_world_state_manager(manager):
    """Set the world state manager instance"""
    global _world_state_manager
    _world_state_manager = manager


@app.get("/", tags=["Root"])
async def root():
    """API root"""
    return {
        "name": "Perception OS",
        "version": "0.1.0",
        "endpoints": [
            "/world_state",
            "/world_state/summary",
            "/world_state/objects/{id}",
            "/health"
        ]
    }


@app.get(
    "/world_state",
    response_model=WorldStateResponse,
    tags=["World State"],
    responses={503: {"model": ErrorResponse}}
)
async def get_world_state():
    """
    Get the complete world state with all tracked objects.
    
    Returns the current perception of the world including:
    - All tracked objects with positions, velocities, and confidence
    - Objects that are occluded but still being predicted
    - Timestamp and frame information
    """
    if _world_state_manager is None:
        raise HTTPException(status_code=503, detail="Perception system not initialized")
    
    state = _world_state_manager.get_world_state()
    if state is None:
        raise HTTPException(status_code=503, detail="No world state available")
    
    return state


@app.get(
    "/world_state/summary",
    response_model=WorldStateSummary,
    tags=["World State"]
)
async def get_world_state_summary():
    """
    Get a summary of the world state.
    
    Quick overview of object counts by state (tracked, occluded, uncertain, lost).
    Useful for monitoring and dashboards.
    """
    if _world_state_manager is None:
        raise HTTPException(status_code=503, detail="Perception system not initialized")
    
    return _world_state_manager.get_summary()


@app.get(
    "/world_state/objects/{object_id}",
    response_model=TrackedObjectSchema,
    tags=["World State"],
    responses={404: {"model": ErrorResponse}}
)
async def get_object(object_id: str):
    """
    Get details for a specific tracked object.
    
    Args:
        object_id: The unique object identifier (e.g., obj_0001)
    
    Returns detailed information about a single tracked object.
    """
    if _world_state_manager is None:
        raise HTTPException(status_code=503, detail="Perception system not initialized")
    
    obj = _world_state_manager.get_object(object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"Object {object_id} not found")
    
    return obj


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"]
)
async def get_health():
    """
    Get system health status.
    
    Returns:
    - Current FPS
    - GPU information
    - Frame count
    - Number of tracked objects
    """
    if _world_state_manager is None:
        return HealthResponse(
            status="initializing",
            fps=0.0,
            frame_count=0,
            gpu="N/A",
            objects_tracked=0
        )
    
    return _world_state_manager.get_health()


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    uvicorn.run(app, host=host, port=port)
