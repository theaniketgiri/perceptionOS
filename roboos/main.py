"""
Main entry point for Perception OS.
Runs the perception loop and API server.
"""

import argparse
import threading
import time
import signal
import sys
from loguru import logger

from roboos.config import config
from roboos.perception.camera import Camera
from roboos.tracking.world_state import WorldStateManager
from roboos.viz.visualizer import Visualizer
from roboos.api.api import app, set_world_state_manager


# Global flag for graceful shutdown
_running = True


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    global _running
    logger.info("Shutdown signal received")
    _running = False


def run_perception_loop(
    world_state_manager: WorldStateManager,
    camera: Camera,
    visualizer: Visualizer = None,
    headless: bool = False
):
    """
    Main perception loop.
    
    Args:
        world_state_manager: World state manager instance
        camera: Camera instance
        visualizer: Optional visualizer for display
        headless: If True, run without visualization
    """
    global _running
    
    logger.info("Starting perception loop")
    
    with camera:
        while _running:
            # Read frame
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame
            world_state = world_state_manager.process_frame(frame)
            
            # Visualize if enabled
            if not headless and visualizer is not None:
                objects = world_state.objects
                fps = world_state_manager.fps
                
                if not visualizer.show(frame.image, objects, fps):
                    logger.info("Visualization closed by user")
                    _running = False
                    break
    
    if visualizer is not None:
        visualizer.close()
    
    logger.info("Perception loop stopped")


def run_api_server(host: str, port: int):
    """Run the FastAPI server"""
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Perception OS")
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default="0",
        help="Camera source (0 for webcam, or RTSP URL)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API server"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="API server port"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Banner
    logger.info("=" * 50)
    logger.info("  PERCEPTION OS v0.1.0")
    logger.info("  Real-time world state perception")
    logger.info("=" * 50)
    
    # Parse camera source
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    
    # Override device
    config.model.device = args.device
    
    # Initialize components
    logger.info("Initializing components...")
    
    camera = Camera(source=camera_source)
    world_state_manager = WorldStateManager()
    visualizer = None if args.headless else Visualizer()
    
    # Initialize world state manager
    if not world_state_manager.initialize():
        logger.error("Failed to initialize perception system")
        sys.exit(1)
    
    # Set world state manager for API
    set_world_state_manager(world_state_manager)
    
    # Start API server in background thread
    api_thread = None
    if not args.no_api:
        api_thread = threading.Thread(
            target=run_api_server,
            args=(config.api.host, args.port),
            daemon=True
        )
        api_thread.start()
        logger.info(f"API available at http://localhost:{args.port}")
    
    # Run perception loop (blocks until shutdown)
    try:
        run_perception_loop(
            world_state_manager=world_state_manager,
            camera=camera,
            visualizer=visualizer,
            headless=args.headless
        )
    except Exception as e:
        logger.exception(f"Error in perception loop: {e}")
    finally:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
