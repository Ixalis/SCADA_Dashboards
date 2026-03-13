#!/usr/bin/env python3
"""
SCADA Detector Calibration Script
==================================
Run this on your Mac at: /Users/ixa/Phase_2/scada_detector/

This script:
1. Generates normal simulator data
2. Trains/calibrates an Isolation Forest specifically for simulator distributions  
3. Validates detection on each attack type
4. Saves calibrated model as isolation_forest_sim_calibrated.pkl
5. Prints recommended thresholds for routes.py

Usage:
    cd /Users/ixa/Phase_2/scada_detector
    python calibrate_simulator.py
"""

import numpy as np
import joblib
import os
import sys

# Ensure we can import the simulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.simulator.data_generator import SCADASimulator, AttackType
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("SCADA Detector — Simulator Calibration")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate normal training data from simulator
    # ------------------------------------------------------------------
    print("\n[1/5] Generating normal simulator data (3000 samples)...")
    sim = SCADASimulator(n_features=51, seed=42)
    normal_data = np.array([sim.generate_sample()[0] for _ in range(3000)])
    print(f"  Shape: {normal_data.shape}")

    # ------------------------------------------------------------------
    # Step 2: Fit scaler on simulator data
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting StandardScaler on simulator data...")
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_data)
    print(f"  Done. Feature means range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")

    # ------------------------------------------------------------------
    # Step 3: Train Isolation Forest
    # ------------------------------------------------------------------
    print("\n[3/5] Training Isolation Forest (300 trees, max_samples=128)...")
    if_model = IsolationForest(
        n_estimators=300,
        max_samples=128,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(normal_scaled)

    normal_scores = if_model.decision_function(normal_scaled)
    p01 = np.percentile(normal_scores, 1)
    p02 = np.percentile(normal_scores, 2)
    p05 = np.percentile(normal_scores, 5)

    print(f"  Normal score stats:")
    print(f"    Mean:   {normal_scores.mean():.6f}")
    print(f"    Std:    {normal_scores.std():.6f}")
    print(f"    Range:  [{normal_scores.min():.6f}, {normal_scores.max():.6f}]")
    print(f"  Threshold candidates:")
    print(f"    P01 (strict, ~1% FP):    {p01:.6f}")
    print(f"    P02 (balanced, ~2% FP):  {p02:.6f}")
    print(f"    P05 (sensitive, ~5% FP): {p05:.6f}")

    # ------------------------------------------------------------------
    # Step 4: Validate with attacks
    # ------------------------------------------------------------------
    print("\n[4/5] Validating attack detection...\n")

    attacks = [
        (AttackType.POINT_SPIKE,       "Point Spike",     0, 10.0),
        (AttackType.POINT_SPIKE,       "Big Spike",       0, 50.0),
        (AttackType.SLOW_DRIFT,        "Slow Drift",      0, 10.0),
        (AttackType.NOISE_INJECTION,   "Noise Injection",  0, 10.0),
        (AttackType.SCALING,           "Scaling",          0, 10.0),
        (AttackType.FROZEN,            "Frozen Sensor",    0, 1.0),
        (AttackType.CORRELATION_BREAK, "Corr Break",       0, 1.0),
        (AttackType.REPLAY,            "Replay",           0, 1.0),
    ]

    print(f"  {'Attack':<20s} {'Det@P05':>8s} {'Det@P01':>8s} {'Min Score':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")

    for atype, name, sensor, intensity in attacks:
        sim_t = SCADASimulator(n_features=51, seed=99)
        # Warm up to build history
        for _ in range(200):
            sim_t.generate_sample()

        sim_t.start_attack(atype, duration_samples=50, target_sensor=sensor, intensity=intensity)

        scores = []
        for _ in range(50):
            sample, _, _ = sim_t.generate_sample()
            scaled = scaler.transform(sample.reshape(1, -1))
            scores.append(if_model.decision_function(scaled)[0])
        scores = np.array(scores)

        det05 = (scores < p05).sum()
        det01 = (scores < p01).sum()
        print(f"  {name:<20s} {det05:>3d}/50   {det01:>3d}/50   {scores.min():>10.4f}")

    # ------------------------------------------------------------------
    # Step 5: Save calibrated model
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving calibrated model...")

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saved_models",
        "isolation_forest_sim_calibrated.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    calibrated = {
        "model": if_model,
        "scaler": scaler,
        "threshold": p05,
        "threshold_strict": p01,
        "sensor_names": [f"sensor_{i}" for i in range(51)],
        "n_features": 51,
        "calibration_source": "simulator",
        "calibration_samples": 3000,
    }
    joblib.dump(calibrated, save_path)
    print(f"  Saved to: {save_path}")

    # ------------------------------------------------------------------
    # Print instructions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HOW TO USE")
    print("=" * 60)
    print(f"""
Option A — Use the calibrated model file:
  python main.py \\
    --if-model saved_models/isolation_forest_sim_calibrated.pkl \\
    --transformer-model /Users/ixa/Phase_2/data/Models/lstm_ae_best.keras \\
    --scaler /Users/ixa/Phase_2/data/Models/scaler.pkl

Option B — Keep the SWaT model but override thresholds at runtime:
  After starting the server, call:
    curl -X POST http://localhost:8000/thresholds \\
      -H "Content-Type: application/json" \\
      -d '{{"if_threshold": {p05:.6f}, "transformer_threshold": 0.5}}'

  NOTE: Option B won't work well because the SWaT scaler maps simulator
  data into a region where all scores cluster together. Option A is
  strongly recommended.

What to expect on the dashboard:
  - Normal operation: ~95% green, ~5% yellow (expected FP rate)
  - Point attacks on single sensor: ~25-30% detected by IF
  - Correlation break / noise: ~40-50% detected by IF
  - Slow drift / replay: Low IF detection — Transformer handles these
  - Combined ensemble: Much higher detection once Transformer buffer fills
""")


if __name__ == "__main__":
    main()
