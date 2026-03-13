"""
SCADA Anomaly Detection Server - Configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    # Paths (update these for your system)
    isolation_forest_path: str = "saved_models/isolation_forest.pkl"
    transformer_model_path: str = "saved_models/transformer_ae.keras"
    transformer_scaler_path: str = "saved_models/transformer_scaler.pkl"
    
    # Transformer settings
    sequence_length: int = 100
    n_features: int = 51  # SWaT has 51 sensors
    
    # Thresholds (from your optimization)
    if_threshold: float = -0.05
    transformer_threshold: float = 0.278
    
    # Ensemble mode: 'or', 'and', 'majority', 'weighted'
    ensemble_mode: str = "or"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Buffer settings
    buffer_max_size: int = 1000  # Max samples to keep
    
    # Alert settings
    alert_cooldown_seconds: float = 5.0  # Don't spam alerts
    webhook_url: Optional[str] = None  # Optional webhook for alerts


@dataclass
class SimulatorConfig:
    """Built-in simulator configuration."""
    # Sensor names (simplified for demo, can expand to full SWaT)
    sensor_names: List[str] = field(default_factory=lambda: [
        "LIT101", "LIT301", "LIT401",  # Level sensors
        "FIT101", "FIT201", "FIT301", "FIT401", "FIT501", "FIT601",  # Flow
        "AIT201", "AIT202", "AIT203", "AIT401", "AIT402",  # Analyzers
        "AIT501", "AIT502", "AIT503", "AIT504",
        "PIT501", "PIT502", "PIT503",  # Pressure
        "P101", "P102", "P201", "P202", "P301", "P302",  # Pumps (binary)
        "P401", "P402", "P501", "P502", "P601", "P602",
        "MV101", "MV201", "MV301", "MV302", "MV303", "MV304",  # Valves
        "UV401", "P203", "P204", "P205", "P206",
        "sensor_41", "sensor_42", "sensor_43", "sensor_44",
        "sensor_45", "sensor_46", "sensor_47", "sensor_48",
        "sensor_49", "sensor_50", "sensor_51"
    ])
    
    # Simulation speed (samples per second)
    sample_rate: float = 10.0
    
    # Attack injection probability (for demo)
    attack_probability: float = 0.05


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


# Global config instance
config = Config()


def update_config_for_deployment(
    if_path: str = None,
    transformer_path: str = None,
    scaler_path: str = None,
    n_features: int = None
):
    """Update config for specific deployment."""
    if if_path:
        config.model.isolation_forest_path = if_path
    if transformer_path:
        config.model.transformer_model_path = transformer_path
    if scaler_path:
        config.model.transformer_scaler_path = scaler_path
    if n_features:
        config.model.n_features = n_features
