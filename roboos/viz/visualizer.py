"""
Real-time visualization for Perception OS.
Displays tracked objects with bounding boxes, IDs, and confidence.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


# Color palette for different states
COLORS = {
    "tracked": (0, 255, 0),      # Green
    "occluded": (255, 165, 0),   # Orange
    "uncertain": (255, 255, 0),  # Yellow
    "lost": (128, 128, 128),     # Gray
}

# Class-specific colors (for variety)
CLASS_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
]


class Visualizer:
    """Real-time visualization of tracked objects"""
    
    def __init__(
        self,
        window_name: str = "Perception OS",
        show_confidence: bool = True,
        show_velocity: bool = True,
        show_predictions: bool = True
    ):
        self.window_name = window_name
        self.show_confidence = show_confidence
        self.show_velocity = show_velocity
        self.show_predictions = show_predictions
        
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._thickness = 2
    
    def draw(
        self,
        frame: np.ndarray,
        objects: List[dict],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draw visualization on frame.
        
        Args:
            frame: Input BGR image
            objects: List of tracked objects (from world state)
            fps: Current FPS for display
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw each object
        for obj in objects:
            self._draw_object(output, obj)
        
        # Draw FPS and stats
        self._draw_stats(output, objects, fps)
        
        return output
    
    def _draw_object(self, frame: np.ndarray, obj: dict):
        """Draw a single tracked object"""
        state = obj.get("state", "tracked")
        color = COLORS.get(state, (255, 255, 255))
        
        # Get position
        pos = obj.get("position", [0, 0])
        x, y = int(pos[0]), int(pos[1])
        
        # Draw center point
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 10, color, 2)
        
        # Draw ID label
        obj_id = obj.get("id", "unknown")
        obj_class = obj.get("class", "object")
        label = f"{obj_id}: {obj_class}"
        
        label_pos = (x + 15, y - 10)
        cv2.putText(
            frame, label, label_pos,
            self._font, self._font_scale, color, self._thickness
        )
        
        # Draw confidence bar
        if self.show_confidence:
            confidence = obj.get("confidence", 0.0)
            identity_conf = obj.get("identity_confidence", 0.0)
            
            bar_x = x + 15
            bar_y = y + 5
            bar_width = 60
            bar_height = 8
            
            # Detection confidence
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (100, 100, 100), -1
            )
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_width * confidence), bar_y + bar_height),
                color, -1
            )
            
            # Identity confidence (below)
            bar_y2 = bar_y + bar_height + 2
            cv2.rectangle(
                frame,
                (bar_x, bar_y2),
                (bar_x + bar_width, bar_y2 + bar_height),
                (100, 100, 100), -1
            )
            cv2.rectangle(
                frame,
                (bar_x, bar_y2),
                (bar_x + int(bar_width * identity_conf), bar_y2 + bar_height),
                (200, 200, 0), -1
            )
        
        # Draw velocity vector
        if self.show_velocity:
            velocity = obj.get("velocity", [0, 0])
            vx, vy = velocity[0], velocity[1]
            
            if abs(vx) > 0.5 or abs(vy) > 0.5:
                scale = 5.0  # Scale up for visibility
                end_x = int(x + vx * scale)
                end_y = int(y + vy * scale)
                cv2.arrowedLine(
                    frame, (x, y), (end_x, end_y),
                    (0, 200, 255), 2, tipLength=0.3
                )
        
        # Draw state label for non-tracked
        if state != "tracked":
            state_label = f"[{state.upper()}]"
            cv2.putText(
                frame, state_label, (x + 15, y + 35),
                self._font, 0.4, color, 1
            )
            
            # Time since seen
            last_seen = obj.get("last_seen_ms", 0)
            if last_seen > 0:
                time_label = f"{last_seen}ms ago"
                cv2.putText(
                    frame, time_label, (x + 15, y + 50),
                    self._font, 0.4, (180, 180, 180), 1
                )
    
    def _draw_stats(
        self,
        frame: np.ndarray,
        objects: List[dict],
        fps: float
    ):
        """Draw overall statistics"""
        h, w = frame.shape[:2]
        
        # Background for stats
        cv2.rectangle(frame, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 90), (100, 100, 100), 1)
        
        # Count by state
        counts = {"tracked": 0, "occluded": 0, "uncertain": 0}
        for obj in objects:
            state = obj.get("state", "tracked")
            if state in counts:
                counts[state] += 1
        
        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (20, 30),
            self._font, 0.5, (255, 255, 255), 1
        )
        
        # Object counts
        cv2.putText(
            frame, f"Tracked: {counts['tracked']}", (20, 50),
            self._font, 0.5, COLORS["tracked"], 1
        )
        cv2.putText(
            frame, f"Occluded: {counts['occluded']}", (20, 70),
            self._font, 0.5, COLORS["occluded"], 1
        )
        cv2.putText(
            frame, f"Uncertain: {counts['uncertain']}", (20, 85),
            self._font, 0.4, COLORS["uncertain"], 1
        )
    
    def show(
        self,
        frame: np.ndarray,
        objects: List[dict],
        fps: float = 0.0
    ) -> bool:
        """
        Show visualization in window.
        
        Returns:
            False if user pressed 'q' to quit, True otherwise
        """
        output = self.draw(frame, objects, fps)
        cv2.imshow(self.window_name, output)
        
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def close(self):
        """Close visualization window"""
        cv2.destroyAllWindows()
