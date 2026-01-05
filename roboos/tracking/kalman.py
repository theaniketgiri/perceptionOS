"""
Kalman Filter for 2D object state estimation.

Implements a constant-velocity motion model for tracking objects in pixel space.
Provides predict/update cycle with proper uncertainty propagation.

Mathematical Model:
    State: x = [x, y, vx, vy]^T
    Measurement: z = [x, y]^T
    
    State transition: x_{k+1} = F * x_k + w, w ~ N(0, Q)
    Measurement: z_k = H * x_k + v, v ~ N(0, R)

Design Principles:
    1. Numerically stable: Uses Joseph form for covariance update
    2. Bounded uncertainty: Covariance cannot grow unbounded
    3. Explicit state: No hidden state, everything is inspectable

Author: Perception OS Team
License: MIT
"""

from __future__ import annotations

import numpy as np
from typing import List, Final
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

# State dimensions
STATE_DIM: Final[int] = 4      # [x, y, vx, vy]
MEASUREMENT_DIM: Final[int] = 2  # [x, y]

# Default noise parameters (tuned for ~30 FPS video)
DEFAULT_PROCESS_NOISE: Final[float] = 1.0
DEFAULT_MEASUREMENT_NOISE: Final[float] = 10.0

# Covariance bounds to prevent numerical issues
MAX_COVARIANCE: Final[float] = 10000.0
MIN_COVARIANCE: Final[float] = 0.01


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True, slots=True)
class KalmanState:
    """
    Immutable snapshot of Kalman filter state.
    
    Using frozen dataclass ensures state snapshots cannot be accidentally modified.
    """
    x: float
    y: float
    vx: float
    vy: float
    
    @property
    def position(self) -> np.ndarray:
        """Position as numpy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float64)
    
    @property
    def velocity(self) -> np.ndarray:
        """Velocity as numpy array [vx, vy]."""
        return np.array([self.vx, self.vy], dtype=np.float64)
    
    def to_array(self) -> np.ndarray:
        """Full state as numpy array [x, y, vx, vy]."""
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> KalmanState:
        """Construct from numpy array."""
        if len(arr) != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM} elements, got {len(arr)}")
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            vx=float(arr[2]),
            vy=float(arr[3])
        )
    
    @classmethod
    def zero(cls) -> KalmanState:
        """Construct zero state."""
        return cls(x=0.0, y=0.0, vx=0.0, vy=0.0)


# =============================================================================
# Kalman Filter Implementation
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter for 2D object tracking with constant velocity model.
    
    Usage:
        kf = KalmanFilter()
        kf.initialize(100, 200)  # Start at position (100, 200)
        
        # Each frame:
        predicted_state = kf.predict(dt=1.0)
        updated_state = kf.update(measured_x, measured_y)
    
    The filter maintains:
        - State estimate: [x, y, vx, vy]
        - State covariance: 4x4 matrix representing uncertainty
    
    Thread Safety:
        This class is NOT thread-safe. Use external synchronization
        if accessing from multiple threads.
    """
    
    __slots__ = (
        '_x', '_P', '_F', '_H', '_Q', '_R',
        '_process_noise', '_measurement_noise', '_initialized'
    )
    
    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE
    ) -> None:
        """
        Initialize Kalman filter with noise parameters.
        
        Args:
            process_noise: Process noise scale. Higher = trust measurements more.
            measurement_noise: Measurement noise scale. Higher = trust model more.
        """
        if process_noise <= 0:
            raise ValueError(f"process_noise must be positive, got {process_noise}")
        if measurement_noise <= 0:
            raise ValueError(f"measurement_noise must be positive, got {measurement_noise}")
        
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        
        # State estimate: [x, y, vx, vy]
        self._x = np.zeros(STATE_DIM, dtype=np.float64)
        
        # State covariance matrix
        self._P = np.eye(STATE_DIM, dtype=np.float64) * 100.0
        
        # State transition matrix (will be updated with dt in predict)
        # x' = x + vx*dt
        # y' = y + vy*dt
        # vx' = vx
        # vy' = vy
        self._F = np.eye(STATE_DIM, dtype=np.float64)
        
        # Measurement matrix: we observe [x, y]
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        
        # Process noise covariance
        self._Q = np.eye(STATE_DIM, dtype=np.float64) * process_noise
        
        # Measurement noise covariance
        self._R = np.eye(MEASUREMENT_DIM, dtype=np.float64) * measurement_noise
        
        self._initialized = False
    
    def initialize(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        initial_covariance: float = 100.0
    ) -> None:
        """
        Initialize filter with known position (and optional velocity).
        
        Args:
            x: Initial x position
            y: Initial y position
            vx: Initial x velocity (default 0)
            vy: Initial y velocity (default 0)
            initial_covariance: Initial uncertainty (default 100)
        """
        self._x = np.array([x, y, vx, vy], dtype=np.float64)
        self._P = np.eye(STATE_DIM, dtype=np.float64) * initial_covariance
        self._initialized = True
    
    def predict(self, dt: float = 1.0) -> KalmanState:
        """
        Predict state at next timestep.
        
        Args:
            dt: Time delta (in frames or seconds, must be consistent)
            
        Returns:
            Predicted state (filter internal state is also updated)
            
        Raises:
            RuntimeError: If filter not initialized
        """
        if not self._initialized:
            raise RuntimeError("Kalman filter not initialized. Call initialize() first.")
        
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # Update state transition matrix with time step
        F = self._F.copy()
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt
        
        # Predict state: x = F @ x
        self._x = F @ self._x
        
        # Predict covariance: P = F @ P @ F^T + Q
        # Scale process noise by dt
        Q_scaled = self._Q * dt
        self._P = F @ self._P @ F.T + Q_scaled
        
        # Enforce covariance bounds for numerical stability
        self._clamp_covariance()
        
        return KalmanState.from_array(self._x)
    
    def update(self, z_x: float, z_y: float) -> KalmanState:
        """
        Update state with a measurement.
        
        Args:
            z_x: Measured x position
            z_y: Measured y position
            
        Returns:
            Updated state
            
        Note:
            If not initialized, this will initialize the filter with
            the measurement position and zero velocity.
        """
        if not self._initialized:
            self.initialize(z_x, z_y)
            return KalmanState.from_array(self._x)
        
        z = np.array([z_x, z_y], dtype=np.float64)
        
        # Innovation (measurement residual): y = z - H @ x
        y = z - self._H @ self._x
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = self._H @ self._P @ self._H.T + self._R
        
        # Kalman gain: K = P @ H^T @ S^(-1)
        # Use solve instead of inverse for numerical stability
        try:
            K = np.linalg.solve(S.T, (self._P @ self._H.T).T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            K = self._P @ self._H.T @ np.linalg.pinv(S)
        
        # Update state: x = x + K @ y
        self._x = self._x + K @ y
        
        # Update covariance using Joseph form for numerical stability
        # P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        I_KH = np.eye(STATE_DIM) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T
        
        # Enforce symmetry (can drift due to numerical errors)
        self._P = (self._P + self._P.T) / 2.0
        
        # Enforce covariance bounds
        self._clamp_covariance()
        
        return KalmanState.from_array(self._x)
    
    def predict_n_steps(self, n: int, dt: float = 1.0) -> List[KalmanState]:
        """
        Predict n steps into the future WITHOUT modifying internal state.
        
        Useful for visualization of predicted trajectories.
        
        Args:
            n: Number of steps to predict
            dt: Time delta per step
            
        Returns:
            List of predicted states
        """
        if not self._initialized:
            return [KalmanState.zero() for _ in range(n)]
        
        if n <= 0:
            return []
        
        predictions: List[KalmanState] = []
        
        # Work with copy to avoid modifying internal state
        x = self._x.copy()
        
        F = self._F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        
        for _ in range(n):
            x = F @ x
            predictions.append(KalmanState.from_array(x))
        
        return predictions
    
    def _clamp_covariance(self) -> None:
        """Clamp covariance diagonal to prevent numerical issues."""
        np.clip(
            np.diag(self._P),
            MIN_COVARIANCE,
            MAX_COVARIANCE,
            out=self._P[np.diag_indices_from(self._P)]
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def state(self) -> KalmanState:
        """Current state estimate."""
        if not self._initialized:
            return KalmanState.zero()
        return KalmanState.from_array(self._x)
    
    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance matrix (copy)."""
        return self._P.copy()
    
    @property
    def position_uncertainty(self) -> float:
        """
        Scalar uncertainty of position estimate.
        
        Computed as sqrt of trace of position submatrix of covariance.
        """
        if not self._initialized:
            return float('inf')
        return float(np.sqrt(self._P[0, 0] + self._P[1, 1]))
    
    @property
    def velocity_uncertainty(self) -> float:
        """
        Scalar uncertainty of velocity estimate.
        """
        if not self._initialized:
            return float('inf')
        return float(np.sqrt(self._P[2, 2] + self._P[3, 3]))
    
    @property
    def is_initialized(self) -> bool:
        """True if filter has been initialized with a measurement."""
        return self._initialized
    
    def __repr__(self) -> str:
        if not self._initialized:
            return "KalmanFilter(uninitialized)"
        s = self.state
        return (
            f"KalmanFilter(pos=[{s.x:.1f}, {s.y:.1f}], "
            f"vel=[{s.vx:.1f}, {s.vy:.1f}], "
            f"uncertainty={self.position_uncertainty:.1f})"
        )
