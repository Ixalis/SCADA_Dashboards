"""
Sequence Buffer Manager
-----------------------
Maintains a rolling window of sensor readings for Transformer inference.
Thread-safe for concurrent access.
"""

import numpy as np
from collections import deque
from threading import Lock
from typing import Optional, Tuple
import time


class SequenceBuffer:
    """
    Thread-safe rolling buffer for sensor data.
    
    Maintains a fixed-size window of the most recent samples,
    enabling real-time sequence-based inference.
    """
    
    def __init__(self, sequence_length: int, n_features: int, max_history: int = 1000):
        """
        Args:
            sequence_length: Window size for Transformer
            n_features: Number of sensor features
            max_history: Maximum samples to keep for analysis
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.max_history = max_history
        
        # Current sequence window
        self.window = deque(maxlen=sequence_length)
        
        # Extended history for analysis
        self.history = deque(maxlen=max_history)
        
        # Timestamps
        self.timestamps = deque(maxlen=max_history)
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self.total_samples = 0
        self.start_time = time.time()
    
    def add_sample(self, sample: np.ndarray, timestamp: float = None) -> bool:
        """
        Add a new sample to the buffer.
        
        Args:
            sample: Sensor values (n_features,)
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            True if window is now full and ready for inference
        """
        if len(sample) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(sample)}")
        
        ts = timestamp or time.time()
        
        with self._lock:
            self.window.append(sample.astype(np.float32))
            self.history.append(sample.astype(np.float32))
            self.timestamps.append(ts)
            self.total_samples += 1
        
        return len(self.window) >= self.sequence_length
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence window.
        
        Returns:
            Sequence array (sequence_length, n_features) or None if not ready
        """
        with self._lock:
            if len(self.window) < self.sequence_length:
                return None
            return np.array(list(self.window), dtype=np.float32)
    
    def get_latest_sample(self) -> Optional[np.ndarray]:
        """Get the most recent sample."""
        with self._lock:
            if len(self.window) == 0:
                return None
            return np.array(self.window[-1], dtype=np.float32)
    
    def get_history(self, n: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get historical samples and timestamps.
        
        Args:
            n: Number of samples (None = all available)
            
        Returns:
            (samples array, timestamps array)
        """
        with self._lock:
            if n is None:
                n = len(self.history)
            n = min(n, len(self.history))
            
            samples = np.array(list(self.history)[-n:], dtype=np.float32)
            timestamps = np.array(list(self.timestamps)[-n:])
            
            return samples, timestamps
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data for inference."""
        return len(self.window) >= self.sequence_length
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        elapsed = time.time() - self.start_time
        return {
            'total_samples': self.total_samples,
            'window_size': len(self.window),
            'window_capacity': self.sequence_length,
            'history_size': len(self.history),
            'is_ready': self.is_ready(),
            'samples_per_second': self.total_samples / elapsed if elapsed > 0 else 0,
            'uptime_seconds': elapsed
        }
    
    def clear(self):
        """Clear all data."""
        with self._lock:
            self.window.clear()
            self.history.clear()
            self.timestamps.clear()
            self.total_samples = 0
            self.start_time = time.time()


class MultiSensorBuffer:
    """
    Manages buffers for multiple sensor groups.
    Useful if different subsystems need different models.
    """
    
    def __init__(self):
        self.buffers = {}
    
    def create_buffer(self, name: str, sequence_length: int, n_features: int) -> SequenceBuffer:
        """Create a named buffer."""
        self.buffers[name] = SequenceBuffer(sequence_length, n_features)
        return self.buffers[name]
    
    def get_buffer(self, name: str) -> Optional[SequenceBuffer]:
        """Get buffer by name."""
        return self.buffers.get(name)
    
    def add_sample_all(self, sample: np.ndarray, timestamp: float = None):
        """Add sample to all buffers (if feature count matches)."""
        for buffer in self.buffers.values():
            if len(sample) == buffer.n_features:
                buffer.add_sample(sample, timestamp)
