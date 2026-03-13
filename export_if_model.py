#!/usr/bin/env python3
"""
Export Isolation Forest model for SCADA Server
===============================================

Run this script in your Phase_2 directory to export the IF model
in a format compatible with the SCADA detection server.

Usage:
    python export_if_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# Paths - UPDATE THESE FOR YOUR SYSTEM
NORMAL_PATH = '/Users/ixa/Phase_2/data/Physical/SWaT_Dataset_Normal_v1.xlsx'
OUTPUT_PATH = '/Users/ixa/Phase_2/scada_detector/saved_models/isolation_forest.pkl'

def main():
    print("="*60)
    print("Exporting Isolation Forest Model")
    print("="*60)
    
    # Load normal data
    print("\nLoading normal data...")
    normal_df = pd.read_excel(NORMAL_PATH, header=1)
    normal_df.columns = normal_df.columns.str.strip()
    
    # Get sensor columns
    exclude = ['Timestamp', 'timestamp', 'Normal/Attack', 'label', 'Label', ' Timestamp']
    sensors = [c for c in normal_df.columns if c not in exclude]
    sensors = normal_df[sensors].select_dtypes(include=[np.number]).columns.tolist()
    
    X_normal = normal_df[sensors].ffill().bfill().values.astype(np.float32)
    print(f"Data shape: {X_normal.shape}")
    
    # Fit scaler
    print("\nFitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    
    # Fit Isolation Forest
    print("Fitting Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Compute threshold
    print("Computing threshold...")
    scores = model.decision_function(X_scaled)
    threshold = np.percentile(scores, 5)  # 5th percentile
    
    print(f"Threshold: {threshold:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    export_data = {
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'sensor_names': sensors,
        'n_features': len(sensors)
    }
    
    joblib.dump(export_data, OUTPUT_PATH)
    print(f"\nSaved to: {OUTPUT_PATH}")
    
    # Test
    print("\nVerifying...")
    loaded = joblib.load(OUTPUT_PATH)
    test_sample = X_normal[0:1]
    test_scaled = loaded['scaler'].transform(test_sample)
    test_score = loaded['model'].decision_function(test_scaled)[0]
    print(f"Test score: {test_score:.4f}")
    print(f"Is anomaly: {test_score < loaded['threshold']}")
    
    print("\n✓ Export complete!")
    print(f"  Model: {OUTPUT_PATH}")
    print(f"  Features: {len(sensors)}")
    print(f"  Threshold: {threshold:.4f}")


if __name__ == '__main__':
    main()
