# Perception OS

A real-time perception layer for robotics that maintains a live, latent understanding of the physical world.

## Features

- **Persistent Object Tracking**: Maintain object identity through occlusions
- **Confidence Scoring**: Explicit uncertainty on all predictions
- **Real-time API**: `GET /world_state` for instant world understanding
- **Hybrid Architecture**: V-JEPA 2 for semantics + Kalman filter for kinematics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the perception system
python -m roboos.main

# In another terminal, query the world state
curl http://localhost:8000/world_state
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /world_state` | Full world state with all tracked objects |
| `GET /world_state/summary` | Quick summary (counts by state) |
| `GET /world_state/objects/{id}` | Single object details |
| `GET /health` | System health and FPS |

## Architecture

```
Camera → V-JEPA 2 Encoder → Object Detection → Multi-Object Tracker → World State → API
                                    ↓                    ↑
                              Kalman Filter ─────────────┘
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (RTX 3050 or better)
- Webcam or RTSP camera

## License

MIT
