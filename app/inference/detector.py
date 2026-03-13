"""
SCADA Anomaly Detector
"""
import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Add path for custom layers
sys.path.append('/Users/ixa/Phase_2/models_phase2')
from LSTM_AE_Phase2_FINAL import AttentionLayer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = 0
    LOW = 1      # Single model triggered, low confidence
    MEDIUM = 2   # Single model triggered, high confidence
    HIGH = 3     # Multiple models triggered
    CRITICAL = 4 # All models triggered + sustained


@dataclass
class DetectionResult:
    """Result from a single detection."""
    timestamp: float
    alert_level: AlertLevel
    is_anomaly: bool
    
    # Individual model results
    if_score: float = 0.0
    if_anomaly: bool = False
    transformer_error: float = 0.0
    transformer_anomaly: bool = False
    
    # Ensemble
    ensemble_score: float = 0.0
    detection_source: str = "none"
    
    # Performance
    latency_ms: float = 0.0
    
    # Additional context
    anomalous_sensors: List[str] = field(default_factory=list)
    message: str = ""


class SCADADetector:
    """
    Real-time SCADA anomaly detector.
    
    Combines:
    - Isolation Forest (statistical outliers)
    - Transformer Autoencoder (temporal patterns)
    
    With optional:
    - Rule-based checks (physical limits)
    - Alert persistence (sustained anomaly detection)
    """
    
    def __init__(self, config=None):
        """
        Initialize detector.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config
        
        # Models (loaded later)
        self.isolation_forest = None
        self.transformer = None
        self.scaler = None
        
        # Thresholds
        self.if_threshold = -0.05
        self.transformer_threshold = 0.278
        
        # Sequence buffer
        self.sequence_buffer = deque(maxlen=100)
        self.sequence_length = 100
        self.n_features = 51
        
        # Alert state
        self.consecutive_anomalies = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5.0
        
        # Statistics
        self.total_detections = 0
        self.total_anomalies = 0
        self.latencies = deque(maxlen=1000)
        
        # Model status
        self.if_loaded = False
        self.transformer_loaded = False
        
        logger.info("SCADADetector initialized")

    def load_isolation_forest(self, model_path: str):
        """Load Isolation Forest model."""
        try:
            import joblib
            loaded = joblib.load(model_path)
            
            # Handle if saved as dict
            if isinstance(loaded, dict):
                self.isolation_forest = loaded['model']
                # Also get the scaler if present and we don't have one
                if 'scaler' in loaded and self.scaler is None:
                    self.scaler = loaded['scaler']
                if 'threshold' in loaded:
                    self.if_threshold = loaded['threshold']
                logger.info(f"IF loaded from dict with keys: {loaded.keys()}")
            else:
                self.isolation_forest = loaded
            
            self.if_loaded = True
            logger.info(f"Isolation Forest loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Isolation Forest: {e}")
            self.if_loaded = False    
    
    def load_transformer(self, model_path: str, scaler_path: str):
        """Load Transformer model."""
        try:
            import tensorflow as tf
            import joblib
        
            # Get all custom objects (includes PositionalEncoding, TransformerEncoderBlock, etc.)
            custom_objects = self._get_transformer_custom_objects()
        
            # Add AttentionLayer too
            custom_objects['AttentionLayer'] = AttentionLayer
        
            # Load with ALL custom objects
            self.transformer = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects
            )
        
            self.scaler = joblib.load(scaler_path)
        
            # Get dimensions from model
            self.sequence_length = self.transformer.input_shape[1]
            self.n_features = self.transformer.input_shape[2]
        
            # Reset buffer with correct size
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            self.transformer_loaded = True
            logger.info(f"Transformer loaded: seq_len={self.sequence_length}, features={self.n_features}")
        
        except Exception as e:
            logger.error(f"Failed to load Transformer: {e}")
            self.transformer_loaded = False
    
    def _get_transformer_custom_objects(self):
        """Get custom layer definitions for model loading."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        class PositionalEncoding(layers.Layer):
            def __init__(self, sequence_length, d_model, **kwargs):
                super().__init__(**kwargs)
                self.sequence_length = sequence_length
                self.d_model = d_model
            
            def build(self, input_shape):
                positions = np.arange(self.sequence_length)[:, np.newaxis]
                dimensions = np.arange(self.d_model)[np.newaxis, :]
                angles = positions / np.power(10000, (2 * (dimensions // 2)) / self.d_model)
                angles[:, 0::2] = np.sin(angles[:, 0::2])
                angles[:, 1::2] = np.cos(angles[:, 1::2])
                self.pos_encoding = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
            
            def call(self, x):
                return x + self.pos_encoding[:, :tf.shape(x)[1], :]
            
            def get_config(self):
                config = super().get_config()
                config.update({'sequence_length': self.sequence_length, 'd_model': self.d_model})
                return config
        
        class TransformerEncoderBlock(layers.Layer):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_ff = d_ff
                self.dropout_rate = dropout
            
            def build(self, input_shape):
                self.mha = layers.MultiHeadAttention(
                    num_heads=self.n_heads, 
                    key_dim=self.d_model // self.n_heads, 
                    dropout=self.dropout_rate
                )
                self.ffn = keras.Sequential([
                    layers.Dense(self.d_ff, activation='gelu'),
                    layers.Dropout(self.dropout_rate),
                    layers.Dense(self.d_model),
                    layers.Dropout(self.dropout_rate)
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(self.dropout_rate)
            
            def call(self, x, training=False):
                norm_x = self.layernorm1(x)
                attn_output = self.mha(norm_x, norm_x, training=training)
                attn_output = self.dropout1(attn_output, training=training)
                x = x + attn_output
                norm_x = self.layernorm2(x)
                ffn_output = self.ffn(norm_x, training=training)
                return x + ffn_output
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    'd_model': self.d_model, 
                    'n_heads': self.n_heads, 
                    'd_ff': self.d_ff, 
                    'dropout': self.dropout_rate
                })
                return config
        
        class TransformerDecoderBlock(layers.Layer):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_ff = d_ff
                self.dropout_rate = dropout
            
            def build(self, input_shape):
                self.mha = layers.MultiHeadAttention(
                    num_heads=self.n_heads, 
                    key_dim=self.d_model // self.n_heads, 
                    dropout=self.dropout_rate
                )
                self.ffn = keras.Sequential([
                    layers.Dense(self.d_ff, activation='gelu'),
                    layers.Dropout(self.dropout_rate),
                    layers.Dense(self.d_model),
                    layers.Dropout(self.dropout_rate)
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(self.dropout_rate)
            
            def call(self, x, training=False):
                norm_x = self.layernorm1(x)
                attn_output = self.mha(norm_x, norm_x, training=training)
                attn_output = self.dropout1(attn_output, training=training)
                x = x + attn_output
                norm_x = self.layernorm2(x)
                ffn_output = self.ffn(norm_x, training=training)
                return x + ffn_output
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    'd_model': self.d_model, 
                    'n_heads': self.n_heads, 
                    'd_ff': self.d_ff, 
                    'dropout': self.dropout_rate
                })
                return config
        
        return {
            'PositionalEncoding': PositionalEncoding,
            'TransformerEncoderBlock': TransformerEncoderBlock,
            'TransformerDecoderBlock': TransformerDecoderBlock
        }
    
    def set_thresholds(self, if_threshold: float = None, transformer_threshold: float = None):
        """Update detection thresholds."""
        if if_threshold is not None:
            self.if_threshold = if_threshold
        if transformer_threshold is not None:
            self.transformer_threshold = transformer_threshold
        logger.info(f"Thresholds: IF={self.if_threshold}, Transformer={self.transformer_threshold}")
    
    def preprocess(self, sample: np.ndarray) -> np.ndarray:
        """Preprocess a sample using the loaded scaler."""
        if self.scaler is not None:
            scaled = self.scaler.transform(sample.reshape(1, -1))[0]
            return scaled.astype(np.float32)
        return sample.astype(np.float32)
    
    def detect(self, sample: np.ndarray, timestamp: float = None) -> DetectionResult:
        """
        Run anomaly detection on a single sample.
        
        Args:
            sample: Sensor values (n_features,)
            timestamp: Optional timestamp
            
        Returns:
            DetectionResult with all detection information
        """
        start_time = time.time()
        ts = timestamp or time.time()
        
        # Initialize result
        result = DetectionResult(
            timestamp=ts,
            alert_level=AlertLevel.NORMAL,
            is_anomaly=False
        )
        
        # Preprocess
        sample_scaled = self.preprocess(sample)
        
        # Add to sequence buffer
        self.sequence_buffer.append(sample_scaled)
        
        # === Isolation Forest ===
        if self.if_loaded:
            if_input = sample_scaled.reshape(1, -1)
            result.if_score = self.isolation_forest.decision_function(if_input)[0]
            result.if_anomaly = result.if_score < self.if_threshold
        
        # === Transformer ===
        if self.transformer_loaded and len(self.sequence_buffer) >= self.sequence_length:
            sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
            sequence = np.clip(sequence, -5.0, 5.0)  # Clip for stability
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Reconstruct and compute error
            reconstructed = self.transformer.predict(sequence, verbose=0)
            result.transformer_error = np.mean(np.abs(sequence - reconstructed))
            result.transformer_anomaly = result.transformer_error > self.transformer_threshold
        
        # === Ensemble Decision ===
        sources = []
        if result.if_anomaly:
            sources.append("IF")
        if result.transformer_anomaly:
            sources.append("Transformer")
        
        result.is_anomaly = result.if_anomaly or result.transformer_anomaly
        result.detection_source = "+".join(sources) if sources else "none"
        
        # Compute ensemble score (normalized combination)
        if_normalized = max(0, -result.if_score) / 0.5  # Higher = more anomalous
        trans_normalized = result.transformer_error / self.transformer_threshold
        result.ensemble_score = max(if_normalized, trans_normalized)
        
        # === Alert Level ===
        if result.is_anomaly:
            self.consecutive_anomalies += 1
            
            if result.if_anomaly and result.transformer_anomaly:
                if self.consecutive_anomalies > 5:
                    result.alert_level = AlertLevel.CRITICAL
                else:
                    result.alert_level = AlertLevel.HIGH
            elif result.ensemble_score > 1.5:
                result.alert_level = AlertLevel.MEDIUM
            else:
                result.alert_level = AlertLevel.LOW
        else:
            self.consecutive_anomalies = 0
        
        # === Finalize ===
        result.latency_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self.total_detections += 1
        if result.is_anomaly:
            self.total_anomalies += 1
        self.latencies.append(result.latency_ms)
        
        # Generate message
        result.message = self._generate_message(result)
        
        return result
    
    def _generate_message(self, result: DetectionResult) -> str:
        """Generate human-readable alert message."""
        if result.alert_level == AlertLevel.NORMAL:
            return "System operating normally"
        elif result.alert_level == AlertLevel.LOW:
            return f"Minor anomaly detected by {result.detection_source}"
        elif result.alert_level == AlertLevel.MEDIUM:
            return f"Anomaly detected: {result.detection_source} (score: {result.ensemble_score:.2f})"
        elif result.alert_level == AlertLevel.HIGH:
            return f"⚠️ HIGH ALERT: Multiple detectors triggered ({result.detection_source})"
        else:
            return f"🚨 CRITICAL: Sustained anomaly ({self.consecutive_anomalies} consecutive)"
    
    def detect_batch(self, samples: np.ndarray) -> List[DetectionResult]:
        """Run detection on multiple samples."""
        results = []
        for i, sample in enumerate(samples):
            result = self.detect(sample)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'total_detections': self.total_detections,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / self.total_detections if self.total_detections > 0 else 0,
            'avg_latency_ms': np.mean(self.latencies) if self.latencies else 0,
            'max_latency_ms': np.max(self.latencies) if self.latencies else 0,
            'if_loaded': self.if_loaded,
            'transformer_loaded': self.transformer_loaded,
            'buffer_size': len(self.sequence_buffer),
            'buffer_ready': len(self.sequence_buffer) >= self.sequence_length,
            'consecutive_anomalies': self.consecutive_anomalies
        }
    
    def reset(self):
        """Reset detector state."""
        self.sequence_buffer.clear()
        self.consecutive_anomalies = 0
        self.total_detections = 0
        self.total_anomalies = 0
        self.latencies.clear()
        logger.info("Detector reset")


# Singleton instance for global access
_detector_instance = None

def get_detector() -> SCADADetector:
    """Get or create the global detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SCADADetector()
    return _detector_instance
