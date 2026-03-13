# SCADA Anomaly Detection System

Real-time anomaly detection for SCADA/ICS systems using a hybrid Isolation Forest + LSTM-Autoencoder ensemble.

**Capstone Project** — Ho Chi Minh City University of Technology (HCMUT-VNUHCM)  
Computer Engineering, Faculty of Computer Science and Engineering

## Overview

This system detects cyber-attacks on industrial control systems by combining two complementary approaches:

- **Isolation Forest (IF):** Fast point-based statistical anomaly detection (~1-2ms per sample)
- **LSTM-Autoencoder (Transformer):** Temporal pattern analysis that catches attacks IF misses (slow drift, replay, frozen sensors)

The ensemble runs in real-time with a web dashboard for monitoring, alert logging, and attack simulation.

### Key Results

| Metric | SWaT Dataset | Simulator |
|--------|-------------|-----------|
| IF AUC-ROC | 84.13% | N/A (calibrated) |
| Transformer AUC-ROC | 90.22% | N/A |
| Ensemble Recall | 90.9% | — |
| Unique Transformer Detection | 27.7% of attacks | Point Spike: 18.9%, Slow Drift: 24.4% |
| Edge Inference (ESP32-S3) | — | 1-2ms per sample |

## Architecture

```
[SCADA Sensors / Simulator]
        │
        ▼
┌─────────────────────────┐
│   FastAPI Server        │
│  ├── Isolation Forest   │  ← Point anomaly detection
│  ├── LSTM-Autoencoder   │  ← Temporal pattern detection
│  ├── Ensemble Logic     │  ← OR combination
│  └── WebSocket Stream   │  ← Real-time to dashboard
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Web Dashboard         │
│  ├── Detection Scores   │
│  ├── Sensor Values      │
│  ├── Alert Log          │
│  └── Attack Injection   │
└─────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
pip install fastapi uvicorn websockets jinja2 python-multipart
pip install tensorflow scikit-learn joblib numpy
```

### Download Models

Models are not included in the repo (too large). Train them using the provided scripts or contact the author.

Place model files in `saved_models/`:
- `isolation_forest_sim_calibrated.pkl` — Calibrated IF model
- `lstm_ae_sim.keras` — LSTM-AE trained on simulator data
- `scaler_sim.pkl` — StandardScaler for simulator data

### Run the Server

```bash
cd scada_detector

SCADA_IF_THRESHOLD=-0.02 SCADA_TRANSFORMER_THRESHOLD=0.664908 python main.py \
  --if-model saved_models/isolation_forest_sim_calibrated.pkl \
  --transformer-model saved_models/lstm_ae_sim.keras \
  --scaler saved_models/scaler_sim.pkl
```

### Start the Simulator

In a second terminal (no auto-attacks):

```bash
curl -X POST "http://localhost:8000/simulator/start?samples_per_second=10&attack_probability=0"
```

### Open Dashboard

Navigate to [http://localhost:8000](http://localhost:8000)

Use the **Inject Attack** button to test different attack types:
- Point Spike — detected by Transformer (~18.9% anomaly rate)
- Slow Drift — detected by Transformer (~24.4% anomaly rate)
- Frozen Sensor — not detected (limitation)
- Replay Attack — not detected (limitation)
- Noise Injection — marginal detection

## Project Structure

```
scada_detector/
├── main.py                          # Entry point
├── app/
│   ├── api/
│   │   └── routes.py                # FastAPI endpoints + WebSocket
│   ├── inference/
│   │   └── detector.py              # SCADADetector ensemble class
│   └── simulator/
│       └── data_generator.py        # SCADA data simulator
├── templates/
│   └── dashboard.html               # Web dashboard
├── static/                          # CSS/JS assets
├── saved_models/                    # Model files (not in repo)
├── calibrate_simulator.py           # IF calibration script
├── train_lstm_ae_simulator.py       # LSTM-AE training script
├── reports/                         # Biweekly LaTeX reports
└── edge/
    ├── scada_edge_detector.ino      # Arduino sketch for ESP32-S3
    └── scada_if_model.h             # IF model as C header (50 trees)
```

## Edge Deployment

The Isolation Forest was converted to pure C and tested on an OhStem Yolo UNO (ESP32-S3):

- **Inference latency:** 1-2ms per sample
- **Model:** 50 trees, 51 features, 155 KB C header
- **Hardware:** ESP32-S3, 240 MHz dual-core, 8MB PSRAM, 16MB Flash

See `edge/` directory for the Arduino sketch and C model header.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/health` | GET | Health check |
| `/stats` | GET | Detector statistics |
| `/detect` | POST | Single sample detection |
| `/detect/batch` | POST | Batch detection |
| `/simulator/start` | POST | Start simulator |
| `/simulator/stop` | POST | Stop simulator |
| `/simulator/attack` | POST | Inject attack |
| `/thresholds` | POST | Update thresholds |
| `/ws` | WebSocket | Real-time stream |

## Known Limitations

- **Scaler dependency:** Models must be trained on the same data distribution as the deployment environment. SWaT-trained models do not work on simulator data without recalibration.
- **Single-sensor attacks:** Both IF and Transformer struggle to detect attacks that modify only 1 of 51 features — the signal is too subtle in high-dimensional space.
- **Transformer on simulator:** Detection rates are lower on simulator data than on SWaT because the simulator's attack patterns are simpler than real SCADA attacks.

## Training Scripts

### Calibrate IF for Simulator

```bash
python calibrate_simulator.py
```

Generates 3000 normal samples, trains IF (300 trees), saves calibrated model + scaler.

### Train LSTM-AE for Simulator

```bash
python train_lstm_ae_simulator.py
```

Generates 20000 normal samples, trains Bidirectional LSTM-AE with Attention (128→64→32→64), ~24 min on M1 Pro.

## References

- SWaT Dataset: Goh et al., "A Dataset to Support Research in the Design of Secure Water Treatment Systems," 2017
- Isolation Forest: Liu et al., "Isolation Forest," IEEE ICDM, 2008
- OhStem Yolo UNO: https://ohstem.vn/product/yolo-uno/

## Author

Le Tien Phat — Computer Engineering, HCMUT-VNUHCM
