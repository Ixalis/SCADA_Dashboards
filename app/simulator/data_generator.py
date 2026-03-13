"""
SCADA Data Simulator
--------------------
Generates realistic sensor data for testing the detection system.
Can simulate various attack types.
"""

import numpy as np
import time
import threading
from typing import Callable, Optional, List, Dict, Generator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of attacks to simulate."""
    NONE = "none"
    POINT_SPIKE = "point_spike"      # Sudden value change
    SLOW_DRIFT = "slow_drift"        # Gradual increase/decrease
    REPLAY = "replay"                # Replay old data
    FROZEN = "frozen"                # Sensor stuck at value
    NOISE_INJECTION = "noise"        # Add noise to signal
    SCALING = "scaling"              # Scale values up/down
    CORRELATION_BREAK = "correlation_break"  # Break sensor correlations


@dataclass
class SimulatorState:
    """Current state of the simulator."""
    is_running: bool = False
    current_attack: AttackType = AttackType.NONE
    attack_start_time: float = 0
    attack_duration: float = 0
    samples_generated: int = 0
    attacks_injected: int = 0


class SCADASimulator:
    """
    Generates synthetic SCADA data with optional attack injection.
    
    Models a simplified water treatment process similar to SWaT:
    - Tank levels (LIT): Fluctuate based on inflow/outflow
    - Flow rates (FIT): Controlled by pumps and valves
    - Analyzers (AIT): Chemical measurements with noise
    - Pressure (PIT): Correlated with flow
    - Actuators (P, MV): Binary on/off states
    """
    
    def __init__(self, n_features: int = 51, seed: int = 42):
        """
        Args:
            n_features: Number of sensor features to simulate
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        
        # Process state (internal simulation variables)
        self.state = {
            'tank_levels': np.array([500.0, 600.0, 400.0]),  # 3 tanks
            'flow_rates': np.array([2.0, 1.8, 2.2, 1.9, 2.1, 2.0]),  # 6 flows
            'pump_states': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),  # 12 pumps
            'valve_states': np.array([1, 1, 0, 1, 0, 1]),  # 6 valves
        }
        
        # Attack state
        self.attack_type = AttackType.NONE
        self.attack_target_sensor = 0
        self.attack_intensity = 1.0
        self.attack_samples_remaining = 0
        self.replay_buffer = []
        
        # Simulation state
        self.sim_state = SimulatorState()
        self.sample_callback: Optional[Callable] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # History for replay attacks
        self.history = []
        self.max_history = 500
        
        logger.info(f"Simulator initialized with {n_features} features")
    
    def _simulate_process(self) -> np.ndarray:
        """Simulate one timestep of the water treatment process."""
        
        # === Tank Levels (LIT) - Mean-reverting with noise ===
        targets = np.array([500.0, 600.0, 400.0])
        reversion_rate = 0.05
        self.state['tank_levels'] += reversion_rate * (targets - self.state['tank_levels'])
        self.state['tank_levels'] += self.rng.normal(0, 2, 3)
        self.state['tank_levels'] = np.clip(self.state['tank_levels'], 50, 1000)
        
        # === Flow Rates (FIT) - Based on pump states with noise ===
        base_flow = 2.0
        for i in range(min(6, len(self.state['pump_states']))):
            pump_on = self.state['pump_states'][i * 2] if i * 2 < len(self.state['pump_states']) else 0
            self.state['flow_rates'][i] = base_flow * pump_on + self.rng.normal(0, 0.1)
        self.state['flow_rates'] = np.clip(self.state['flow_rates'], 0, 5)
        
        # === Randomly toggle some actuators (pumps/valves) ===
        if self.rng.random() < 0.02:  # 2% chance per timestep
            idx = self.rng.integers(0, len(self.state['pump_states']))
            self.state['pump_states'][idx] = 1 - self.state['pump_states'][idx]
        
        # === Build sensor vector ===
        sensors = []
        
        # Tank levels (3)
        sensors.extend(self.state['tank_levels'])
        
        # Flow rates (6)
        sensors.extend(self.state['flow_rates'])
        
        # Analyzers (9) - Chemical measurements
        analyzers = self.rng.normal([250, 300, 350, 200, 220, 400, 7.0, 450, 480], 
                                     [10, 15, 12, 8, 10, 20, 0.2, 15, 18])
        sensors.extend(analyzers)
        
        # Pressure (3) - Correlated with flow
        pressure = 150 + 20 * np.mean(self.state['flow_rates']) + self.rng.normal(0, 5, 3)
        sensors.extend(pressure)
        
        # Pump states (12)
        sensors.extend(self.state['pump_states'].astype(float))
        
        # Valve states (6)
        sensors.extend(self.state['valve_states'].astype(float))
        
        # Additional sensors to reach n_features
        remaining = self.n_features - len(sensors)
        if remaining > 0:
            additional = self.rng.normal(100, 10, remaining)
            sensors.extend(additional)
        
        return np.array(sensors[:self.n_features], dtype=np.float32)
    
    def _inject_attack(self, sample: np.ndarray) -> np.ndarray:
        """Inject attack into sample if active."""
        
        if self.attack_samples_remaining <= 0:
            return sample
        
        self.attack_samples_remaining -= 1
        attacked = sample.copy()
        target = self.attack_target_sensor
        
        if self.attack_type == AttackType.POINT_SPIKE:
            # Sudden spike
            attacked[target] += self.attack_intensity * 50
            
        elif self.attack_type == AttackType.SLOW_DRIFT:
            # Gradual drift
            progress = 1 - (self.attack_samples_remaining / self.attack_duration_samples)
            attacked[target] += progress * self.attack_intensity * 30
            
        elif self.attack_type == AttackType.FROZEN:
            # Freeze at current value
            if len(self.history) > 0:
                attacked[target] = self.history[-1][target]
            
        elif self.attack_type == AttackType.REPLAY:
            # Replay old data
            if len(self.replay_buffer) > 0:
                replay_idx = len(self.replay_buffer) - self.attack_samples_remaining - 1
                if 0 <= replay_idx < len(self.replay_buffer):
                    attacked = self.replay_buffer[replay_idx].copy()
            
        elif self.attack_type == AttackType.NOISE_INJECTION:
            # Add extra noise
            attacked[target] += self.rng.normal(0, self.attack_intensity * 10)
            
        elif self.attack_type == AttackType.SCALING:
            # Scale values
            attacked[target] *= (1 + 0.1 * self.attack_intensity)
            
        elif self.attack_type == AttackType.CORRELATION_BREAK:
            # Randomize correlated sensors
            if target < 3:  # Tank level
                attacked[target] = self.rng.uniform(100, 900)
        
        return attacked
    
    def generate_sample(self) -> tuple:
        """
        Generate one sample.
        
        Returns:
            (sample array, is_attack boolean, attack_type string)
        """
        # Simulate normal process
        sample = self._simulate_process()
        
        # Store in history
        self.history.append(sample.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Check if attack is active
        is_attack = self.attack_samples_remaining > 0
        attack_type = self.attack_type.value if is_attack else "none"
        
        # Inject attack if active
        if is_attack:
            sample = self._inject_attack(sample)
            self.sim_state.attacks_injected += 1
        
        self.sim_state.samples_generated += 1
        
        return sample, is_attack, attack_type
    
    def start_attack(self, 
                    attack_type: AttackType, 
                    duration_samples: int = 50,
                    target_sensor: int = 0,
                    intensity: float = 1.0):
        """
        Start an attack.
        
        Args:
            attack_type: Type of attack
            duration_samples: How many samples the attack lasts
            target_sensor: Which sensor to target
            intensity: Attack intensity multiplier
        """
        self.attack_type = attack_type
        self.attack_samples_remaining = duration_samples
        self.attack_duration_samples = duration_samples
        self.attack_target_sensor = target_sensor % self.n_features
        self.attack_intensity = intensity
        
        # For replay attack, capture current buffer
        if attack_type == AttackType.REPLAY and len(self.history) >= duration_samples:
            start_idx = len(self.history) - duration_samples - 50
            start_idx = max(0, start_idx)
            self.replay_buffer = self.history[start_idx:start_idx + duration_samples]
        
        logger.info(f"Attack started: {attack_type.value}, duration={duration_samples}, sensor={target_sensor}")
    
    def stop_attack(self):
        """Stop current attack."""
        self.attack_type = AttackType.NONE
        self.attack_samples_remaining = 0
        logger.info("Attack stopped")
    
    def generate_stream(self, 
                       samples_per_second: float = 10,
                       attack_probability: float = 0.01,
                       attack_duration: int = 50) -> Generator:
        """
        Generate continuous stream of samples.
        
        Args:
            samples_per_second: Generation rate
            attack_probability: Probability of starting random attack
            attack_duration: Duration of random attacks
            
        Yields:
            (sample, is_attack, attack_type) tuples
        """
        interval = 1.0 / samples_per_second
        
        while not self._stop_event.is_set():
            start = time.time()
            
            # Maybe start random attack
            if self.attack_samples_remaining == 0 and self.rng.random() < attack_probability:
                attack = self.rng.choice([
                    AttackType.POINT_SPIKE,
                    AttackType.SLOW_DRIFT,
                    AttackType.FROZEN,
                    AttackType.NOISE_INJECTION
                ])
                target = self.rng.integers(0, min(10, self.n_features))
                self.start_attack(attack, attack_duration, target)
            
            # Generate sample
            yield self.generate_sample()
            
            # Sleep to maintain rate
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def start_streaming(self, 
                       callback: Callable,
                       samples_per_second: float = 10,
                       attack_probability: float = 0.01):
        """
        Start background streaming thread.
        
        Args:
            callback: Function called with (sample, is_attack, attack_type)
            samples_per_second: Generation rate
            attack_probability: Random attack probability
        """
        if self.sim_state.is_running:
            logger.warning("Simulator already running")
            return
        
        self.sample_callback = callback
        self._stop_event.clear()
        
        def stream_worker():
            self.sim_state.is_running = True
            for sample, is_attack, attack_type in self.generate_stream(
                samples_per_second, attack_probability
            ):
                if self._stop_event.is_set():
                    break
                if self.sample_callback:
                    self.sample_callback(sample, is_attack, attack_type)
            self.sim_state.is_running = False
        
        self._thread = threading.Thread(target=stream_worker, daemon=True)
        self._thread.start()
        logger.info(f"Streaming started at {samples_per_second} samples/sec")
    
    def stop_streaming(self):
        """Stop background streaming."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.sim_state.is_running = False
        logger.info("Streaming stopped")
    
    def get_state(self) -> dict:
        """Get current simulator state."""
        return {
            'is_running': self.sim_state.is_running,
            'samples_generated': self.sim_state.samples_generated,
            'attacks_injected': self.sim_state.attacks_injected,
            'current_attack': self.attack_type.value,
            'attack_remaining': self.attack_samples_remaining,
            'n_features': self.n_features
        }
    
    def reset(self):
        """Reset simulator state."""
        self.stop_streaming()
        self.stop_attack()
        self.history.clear()
        self.sim_state = SimulatorState()
        self.state = {
            'tank_levels': np.array([500.0, 600.0, 400.0]),
            'flow_rates': np.array([2.0, 1.8, 2.2, 1.9, 2.1, 2.0]),
            'pump_states': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            'valve_states': np.array([1, 1, 0, 1, 0, 1]),
        }
        logger.info("Simulator reset")


# Quick test
if __name__ == "__main__":
    sim = SCADASimulator(n_features=51)
    
    print("Generating 10 normal samples:")
    for i in range(10):
        sample, is_attack, attack_type = sim.generate_sample()
        print(f"  Sample {i}: shape={sample.shape}, attack={is_attack}")
    
    print("\nStarting point spike attack:")
    sim.start_attack(AttackType.POINT_SPIKE, duration_samples=5, target_sensor=0)
    
    for i in range(10):
        sample, is_attack, attack_type = sim.generate_sample()
        print(f"  Sample {i}: sensor[0]={sample[0]:.1f}, attack={is_attack}, type={attack_type}")
    
    print("\nSimulator state:", sim.get_state())
