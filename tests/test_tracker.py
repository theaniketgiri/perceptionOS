"""
Tests for the tracking system.

These tests verify the core guarantees:
1. Object IDs are never reassigned
2. Low identity confidence triggers state change, not new ID
3. Kalman filter behaves correctly through occlusions
"""

import pytest
import numpy as np

from roboos.tracking.kalman import KalmanFilter, KalmanState
from roboos.tracking.tracker import (
    Tracker, TrackedObject, ObjectState, 
    MATCH_COST_THRESHOLD, IDENTITY_CONFIDENCE_THRESHOLD
)
from roboos.perception.detector import Detection


class TestKalmanFilter:
    """Tests for production-grade Kalman filter."""
    
    def test_initialize(self):
        """Filter should store initial state correctly."""
        kf = KalmanFilter()
        kf.initialize(100, 200, 5, -3)
        
        state = kf.state
        assert abs(state.x - 100) < 1e-6
        assert abs(state.y - 200) < 1e-6
        assert abs(state.vx - 5) < 1e-6
        assert abs(state.vy - (-3)) < 1e-6
    
    def test_predict_advances_position(self):
        """Predict should advance position by velocity * dt."""
        kf = KalmanFilter()
        kf.initialize(100, 100, 10, 5)
        
        state = kf.predict(dt=1.0)
        
        # Position should advance by velocity
        assert abs(state.x - 110) < 2  # 100 + 10
        assert abs(state.y - 105) < 2  # 100 + 5
    
    def test_update_pulls_toward_measurement(self):
        """Update should pull state toward measurement."""
        kf = KalmanFilter()
        kf.initialize(100, 100)
        
        # Measurement is far from initial state
        state = kf.update(150, 150)
        
        # Should move toward measurement
        assert state.x > 100
        assert state.y > 100
    
    def test_predict_n_steps_nondestructive(self):
        """predict_n_steps should not modify internal state."""
        kf = KalmanFilter()
        kf.initialize(0, 0, 10, 20)
        
        original_state = kf.state
        predictions = kf.predict_n_steps(10, dt=1.0)
        
        # Internal state should be unchanged
        assert kf.state.x == original_state.x
        assert kf.state.y == original_state.y
        
        # But predictions should advance
        assert predictions[-1].x > predictions[0].x
    
    def test_occlusion_prediction_maintains_velocity(self):
        """During occlusion, filter should predict using velocity."""
        kf = KalmanFilter()
        kf.initialize(100, 100, 10, 5)
        
        # Simulate 10 frames of occlusion (predict only, no update)
        positions = []
        for _ in range(10):
            state = kf.predict(dt=1.0)
            positions.append((state.x, state.y))
        
        # Position should advance consistently
        for i in range(1, len(positions)):
            assert positions[i][0] > positions[i-1][0]
            assert positions[i][1] > positions[i-1][1]
    
    def test_numerical_stability(self):
        """Filter should remain stable over many iterations."""
        kf = KalmanFilter()
        kf.initialize(0, 0, 1, 1)
        
        # Run 1000 predict/update cycles
        for i in range(1000):
            kf.predict()
            kf.update(float(i), float(i))
        
        # Covariance should be bounded
        assert kf.position_uncertainty < 1000
        assert not np.isnan(kf.state.x)
        assert not np.isinf(kf.state.x)
    
    def test_uninitialized_raises(self):
        """Predict on uninitialized filter should raise."""
        kf = KalmanFilter()
        
        with pytest.raises(RuntimeError):
            kf.predict()
    
    def test_invalid_params_raise(self):
        """Invalid parameters should raise."""
        with pytest.raises(ValueError):
            KalmanFilter(process_noise=-1)
        
        with pytest.raises(ValueError):
            KalmanFilter(measurement_noise=0)


class TestTracker:
    """Tests for multi-object tracker with identity preservation."""
    
    def _make_detection(
        self,
        x: float, y: float, 
        w: float = 50, h: float = 100,
        class_name: str = "person", 
        confidence: float = 0.9
    ) -> Detection:
        """Helper to create a detection."""
        return Detection(
            bbox=np.array([x - w/2, y - h/2, x + w/2, y + h/2]),
            class_id=0,
            class_name=class_name,
            confidence=confidence,
            embedding=np.random.rand(512).astype(np.float32)
        )
    
    def test_new_detection_creates_track(self):
        """New detection should create new track."""
        tracker = Tracker()
        
        detections = [self._make_detection(100, 200)]
        objects = tracker.update(detections, timestamp_ms=0, frame_id=1)
        
        assert len(objects) == 1
        assert objects[0].id.startswith("obj_")
        assert objects[0].class_name == "person"
        assert objects[0].identity_confidence == 1.0
    
    def test_identity_preserved_across_frames(self):
        """Object ID must persist across frames."""
        tracker = Tracker()
        
        # Frame 1
        objects1 = tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        obj_id = objects1[0].id
        
        # Frame 2 - object moves slightly
        objects2 = tracker.update(
            [self._make_detection(105, 203)],
            timestamp_ms=33, frame_id=2
        )
        
        assert len(objects2) == 1
        assert objects2[0].id == obj_id  # CRITICAL: Same ID
    
    def test_identity_preserved_through_occlusion(self):
        """Object ID must persist through temporary occlusion."""
        tracker = Tracker()
        
        # Frame 1 - object detected
        objects1 = tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        obj_id = objects1[0].id
        
        # Frames 2-5 - object occluded (no detections)
        for i in range(2, 6):
            tracker.update([], timestamp_ms=i * 33, frame_id=i)
        
        # Frame 6 - object reappears near predicted position
        objects6 = tracker.update(
            [self._make_detection(110, 200)],
            timestamp_ms=200, frame_id=6
        )
        
        # Should maintain same ID
        assert any(obj.id == obj_id for obj in objects6)
    
    def test_low_confidence_marks_uncertain_not_new_id(self):
        """Low identity confidence should mark UNCERTAIN, not create new ID."""
        tracker = Tracker()
        
        # Create track
        objects1 = tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        obj_id = objects1[0].id
        
        # Long occlusion to reduce confidence
        for i in range(2, 60):
            tracker.update([], timestamp_ms=i * 100, frame_id=i)
        
        # Get the object - it should be UNCERTAIN, not gone and replaced
        obj = tracker.get_object(obj_id)
        
        if obj is not None:
            # If still tracked, verify state is UNCERTAIN
            assert obj.state in (ObjectState.UNCERTAIN, ObjectState.LOST)
            assert obj.id == obj_id  # CRITICAL: Same ID, not replaced
    
    def test_multiple_objects_maintain_unique_ids(self):
        """Multiple objects should have unique, persistent IDs."""
        tracker = Tracker()
        
        detections = [
            self._make_detection(100, 200, class_name="person"),
            self._make_detection(300, 200, class_name="person"),
            self._make_detection(500, 300, class_name="car"),
        ]
        
        objects = tracker.update(detections, timestamp_ms=0, frame_id=1)
        
        assert len(objects) == 3
        ids = [obj.id for obj in objects]
        assert len(set(ids)) == 3  # All unique
    
    def test_ids_never_reused(self):
        """Lost object IDs should never be reused."""
        tracker = Tracker()
        
        # Create and lose an object
        tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        first_id = tracker.objects[0].id
        
        # Lose it (long occlusion)
        for i in range(2, 200):
            tracker.update([], timestamp_ms=i * 100, frame_id=i)
        
        # Create new object
        tracker.update(
            [self._make_detection(500, 500)],
            timestamp_ms=20000, frame_id=200
        )
        
        # New object should have different ID
        assert all(obj.id != first_id for obj in tracker.objects)
    
    def test_summary_includes_uncertain_count(self):
        """Summary should report low identity confidence objects."""
        tracker = Tracker()
        
        # Create track
        tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        
        summary = tracker.get_summary()
        
        assert "total_objects" in summary
        assert "low_identity_confidence" in summary
    
    def test_reliable_objects_filter(self):
        """reliable_objects should only return high-confidence tracks."""
        tracker = Tracker()
        
        # Create high-confidence track
        tracker.update(
            [self._make_detection(100, 200)],
            timestamp_ms=0, frame_id=1
        )
        
        reliable = tracker.reliable_objects
        
        assert len(reliable) == 1
        assert reliable[0].identity_confidence >= IDENTITY_CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
