# Perception OS

A real-time perception layer for robotics that maintains a live, latent understanding of the physical world.

## Overview

Perception OS provides persistent object tracking with re-identification capabilities, enabling robots to maintain consistent object identities even after prolonged occlusions or camera absences.

### Key Features

- Persistent Object Tracking with Re-ID Gallery for identity resurrection
- Confidence Scoring with explicit uncertainty on all predictions
- Real-time REST API for instant world state access
- Hybrid Encoder Architecture supporting V-JEPA 2 and histogram-based embeddings
- Kalman Filter prediction for smooth trajectory estimation

## Installation

```bash
git clone https://github.com/yourusername/ROBOOS.git
cd ROBOOS
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- PyTorch with CUDA support
- Ultralytics YOLOv8
- HuggingFace Transformers (for V-JEPA)

## Quick Start

```bash
# Run with default histogram encoder (15+ FPS)
python -m roboos.main

# Run with V-JEPA semantic encoder (10-15 FPS, better re-ID)
ENCODER_TYPE=vjepa python -m roboos.main

# Query the world state
curl http://localhost:8000/world_state
```

## Configuration

Environment variables for runtime configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| CAMERA_SOURCE | 0 | Camera index or RTSP URL |
| DEVICE | cuda | Compute device (cuda or cpu) |
| API_PORT | 8000 | REST API server port |
| ENCODER_TYPE | histogram | Encoder type (histogram, vjepa, auto) |
| FILTER_CLASSES | true | Enable class filtering for dynamic objects |

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /world_state | Full world state with all tracked objects |
| GET | /world_state/summary | Summary counts by object state |
| GET | /world_state/objects/{id} | Single object details by ID |
| GET | /health | System health status and FPS metrics |

### Response Example

```json
{
  "timestamp_ms": 1704567890123,
  "frame_id": 1234,
  "objects": [
    {
      "id": "obj_0001",
      "class": "person",
      "state": "tracked",
      "position": [320.5, 240.3],
      "velocity": [1.2, -0.5],
      "confidence": 0.95,
      "identity_confidence": 0.88
    }
  ]
}
```

## Architecture

```
Camera Input
    |
    v
Object Detection (YOLOv8)
    |
    v
Embedding Extraction (Histogram / V-JEPA)
    |
    v
Multi-Object Tracker
    |--- Kalman Filter (motion prediction)
    |--- Re-ID Gallery (identity resurrection)
    |
    v
World State Manager
    |
    v
REST API (/world_state)
```

### Core Components

| Component | Description |
|-----------|-------------|
| Detector | YOLOv8-based object detection with class filtering |
| Encoder | Pluggable embedding extraction (histogram or V-JEPA) |
| Tracker | Multi-object tracker with Hungarian matching |
| Kalman Filter | 2D motion prediction with velocity estimation |
| Re-ID Gallery | Stores embeddings of lost objects for resurrection |
| World State | Centralized object registry with state machine |

### Object States

Objects transition through the following states:

1. TRACKED: Object actively detected and matched
2. OCCLUDED: Object not detected but predicted via Kalman filter
3. UNCERTAIN: Low identity confidence, may be wrong match
4. LOST: Object removed, embedding stored in Re-ID gallery

## Project Structure

```
roboos/
    __init__.py
    config.py           # Configuration dataclasses
    main.py             # Entry point
    perception/
        camera.py       # Camera interface
        detector.py     # YOLOv8 detection
        encoder.py      # V-JEPA and histogram encoders
    tracking/
        tracker.py      # Multi-object tracker
        kalman.py       # Kalman filter implementation
        reid_gallery.py # Re-ID gallery for resurrection
        world_state.py  # World state manager
    api/
        api.py          # FastAPI endpoints
        schemas.py      # Pydantic models
    viz/
        visualizer.py   # OpenCV visualization
```

## Performance

Tested on NVIDIA RTX 3050 Laptop GPU:

| Configuration | FPS | Notes |
|---------------|-----|-------|
| Histogram Encoder | 15+ | Default, recommended for real-time |
| V-JEPA (FP16, 8 frames) | 10-15 | Better semantic quality |

## License

MIT License. See LICENSE file for details.

## References

- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Meta AI, 2025)
- YOLOv8 by Ultralytics
- Hungarian Algorithm for optimal assignment
