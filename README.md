# Automated SCADA/ICS Anomaly Detection — Edge AI Deployment

Real-time anomaly detection for industrial control systems using a hybrid Isolation Forest + Transformer-Autoencoder ensemble, deployed on edge hardware.

**Capstone Project — HK252-DATN-246**
Le Tien Phat (2252600) — Computer Engineering, HCMUT-VNUHCM
Supervisor: Dr. Vo Tuan Binh

## Overview

The system pairs two unsupervised models in an OR-gate ensemble:

- **Isolation Forest (IF):** Scores each 51-feature sample independently. Fast (1–2 ms on ESP32-S3, ~50 ms on Pi 3B) but structurally blind to temporal anomalies.
- **Transformer-Autoencoder (TF-AE):** Reconstructs sliding windows of sensor data using self-attention. Flags anomalies via reconstruction error (MAE). Detects slow drift, frozen sensors, replay attacks, and noise injection.

A live dashboard (FastAPI + WebSocket + Chart.js) streams detection scores, sensor values, and alerts at 1 Hz.

**Key finding:** Across five simulated attack types, IF contributed negligibly to detection. The Transformer-AE carried the full detection burden. IF remains useful as a standalone microcontroller sentry but not as an ensemble contributor for temporal attacks.

### Benchmark results (SWaT dataset)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest* | 99.85% | 99.82% | 99.94% | 99.88% | 99.99% |
| Transformer-AE | 73.01% | 17.16% | 88.82% | 28.76% | 90.22% |
| LSTM-AE (Phase 2) | 44.00% | 16.00% | 89.00% | 28.00% | 83.98% |
| Isolation Forest | 85.94% | 44.80% | 70.86% | 54.90% | 84.99% |

*RF is a supervised upper bound trained on labeled attack data. Not used in deployment.

**Important:** The 90.22% AUC-ROC was benchmarked on the full Transformer-AE model (seq_len=100, d_model=128, 8 heads, 3 layers, dff=512). The dashboard/Pi deployment uses a reduced model (seq_len=50, d_model=64, 4 heads, 2 layers, dff=128) that was not separately benchmarked on SWaT.

## Requirements

### Software

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.12+ breaks TFLite wheel availability on aarch64 |
| scikit-learn | 1.3.x | Must match between training and deployment machines exactly (joblib serialization) |
| TensorFlow/Keras | 2.19 | Training only (M1 Pro Mac or Colab) |
| ai-edge-litert | latest | Pi 3B inference runtime (~5 MB, replaces 280 MB full TF) |
| FastAPI | latest | Detection server |
| uvicorn | latest | ASGI server |
| joblib | latest | Model serialization (version-sensitive) |
| numpy | latest | Array operations |
| jinja2 | latest | Dashboard templating |

Install on the training machine:
```bash
pip install tensorflow scikit-learn fastapi uvicorn websockets jinja2 joblib numpy
```

Install on Raspberry Pi 3B:
```bash
pip install ai-edge-litert scikit-learn fastapi uvicorn jinja2 joblib numpy --break-system-packages
```

**Do not install both `tensorflow` and `ai-edge-litert` on the Pi.** They conflict, and TF is too large for 1 GB RAM.

### Hardware

| Platform | Purpose | Specs |
|---|---|---|
| MacBook Pro M1 Pro (32 GB) | Training, development | TF 2.19 with Metal GPU |
| Raspberry Pi 3B | Full ensemble deployment | ARM Cortex-A53 1.2 GHz, 1 GB RAM, Bullseye 64-bit |
| OhStem Yolo UNO (ESP32-S3) | IF-only proof of concept | 240 MHz dual-core, 512 KB SRAM, 8 MB PSRAM, 16 MB flash |

### Raspberry Pi 3B setup

**OS:** Raspberry Pi OS Bullseye 64-bit (not Bookworm, not Trixie — Python 3.11 required for ai-edge-litert wheels).

**Power supply:** 5V / 2.5A wall adapter via micro-USB is mandatory. USB power banks cause voltage drops during pip install, corrupting the SD card. This happened multiple times during development.

**SD card:** 128 GB A2-rated recommended. Cheaper cards corrupt under sustained writes.

**Swap:** Create a 2 GB swap file to supplement the 1 GB physical RAM:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### ESP32-S3 (Yolo UNO) setup

**Arduino IDE 2.x** with ESP32 board package installed. Required board config:
- Flash Size: 16MB (128Mb)
- Partition Scheme: 16M Flash (3MB APP / 9.9MB FATFS)
- USB CDC On Boot: Enabled (otherwise Serial.println output goes to UART pins, not USB)

**WiFi:** ESP32-S3 supports 2.4 GHz only. 5 GHz networks will not connect.

## Quick start

### 1. Clone and place models

```bash
git clone https://github.com/Ixalis/SCADA_Dashboards.git
cd SCADA_Dashboards/scada_detector
```

Models are not in the repo (too large). Place them in `saved_models/`:

| File | Description | Source |
|---|---|---|
| `isolation_forest_sim_calibrated.pkl` | 50-tree IF, trained on simulator normal data | `calibrate_simulator.py` |
| `lstm_ae_sim_edge.tflite` | Transformer-AE TFLite model (0.7 MB) | Converted from Keras via concrete function |
| `scaler_sim.pkl` | StandardScaler fitted on simulator normal data | Same script as IF |

**scikit-learn version warning:** The `.pkl` files are version-locked. If you trained on scikit-learn 1.3.2, the Pi must also run 1.3.2. Mismatched versions produce `ModuleNotFoundError` or silent corruption.

### 2. Run the server

```bash
SCADA_IF_THRESHOLD=-0.02 \
SCADA_TRANSFORMER_THRESHOLD=0.58 \
python3 main.py \
  --if-model saved_models/isolation_forest_sim_calibrated.pkl \
  --transformer-model saved_models/lstm_ae_sim_edge.tflite \
  --scaler saved_models/scaler_sim.pkl
```

### 3. Open the dashboard

Navigate to `http://localhost:8000` (or `http://<pi-ip>:8000` for remote access).

Use the controls to start the simulator and inject attacks.

## Threshold tuning

Both thresholds are set via environment variables and need recalibration whenever the data source changes.

| Variable | Default | Meaning |
|---|---|---|
| `SCADA_IF_THRESHOLD` | -0.02 | IF anomaly score cutoff (sklearn convention: below = anomaly) |
| `SCADA_TRANSFORMER_THRESHOLD` | 0.58 | Transformer MAE reconstruction error cutoff (above = anomaly) |

**How to calibrate for a new data source:**

1. Run the system on normal data (no attacks) for several hundred samples.
2. Observe the Transformer reconstruction error distribution on the dashboard.
3. Set the threshold above the normal distribution's mass. For SWaT benchmark data, 0.278 worked. For the simulator, distributional mismatch required raising it to 0.58 to eliminate false alarms.
4. For IF, the `contamination` parameter (default 0.05) controls the internal threshold during training. The dashboard threshold provides a secondary cutoff.

**If you see 40–50% anomaly rate during normal operation**, the threshold is too low for your data distribution. Raise it until normal operation produces 0% anomaly rate, then verify attacks still trigger detection.

## Attack types and detection

Tested on the dashboard SCADA simulator:

| Attack | What it does | Transformer | IF |
|---|---|---|---|
| Point Spike | Sudden large offset on one sensor | Crosses threshold | Barely responds |
| Slow Drift | Gradual increment per timestep | Hovers near threshold (6.2% anomaly rate) | Flat |
| Frozen Sensor | Locks sensor to current value | Strongest response (~0.65 peak) | Flat |
| Replay Attack | Replays recorded normal segment | Persistent above threshold | Silent |
| Noise Injection | Adds Gaussian noise to channels | Above threshold | Minor elevation |

**Point spike detection latency:** The sliding window (length 50) needs to fill with anomalous samples before the reconstruction error rises significantly. Expect 1–2 injection cycles before detection registers. This is inherent to the windowed approach, not a bug.

## TFLite model conversion

The Transformer-AE uses layers that internally rely on dynamic TensorList operations. Standard TFLite conversion pulls in the Flex delegate, requiring the full 280 MB TensorFlow runtime — unacceptable on a 1 GB device.

**Solution: concrete function conversion.** Lock the model to a fixed input shape before converting:

```python
import tensorflow as tf

model = tf.keras.models.load_model('transformer_ae.keras')

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 50, 51], dtype=tf.float32)
])
def predict(x):
    return model(x, training=False)

concrete_func = predict.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open('transformer_ae.tflite', 'wb') as f:
    f.write(tflite_model)
```

This eliminates all Flex delegate dependencies. The resulting `.tflite` runs on `ai-edge-litert` (~5 MB) instead of full TensorFlow (~280 MB).

**If you change the sequence length or feature count**, update the `TensorSpec` shape accordingly: `[1, seq_len, num_features]`.

## ESP32-S3 (Isolation Forest only)

The IF model is converted to C via emlearn:

```python
import emlearn
import joblib

model = joblib.load('isolation_forest_sim_calibrated.pkl')
cmodel = emlearn.convert(model, method='inline')
cmodel.save(file='scada_if_model.h', name='scada_if')
```

| Variant | Trees | C header size | ESP32-S3 status |
|---|---|---|---|
| Full model | 300 | 945 KB | Boot crash (memory exhaustion) |
| Reduced model | 50 | 155 KB | Runs at 1–2 ms inference |

The C header includes static arrays for node splits, feature indices, thresholds, and child pointers. No external dependencies required at runtime.

Flash/memory breakdown (50-tree model):

| Component | Size |
|---|---|
| IF model C header | 155 KB |
| Arduino sketch + WiFi stack | ~675 KB |
| Total flash | ~830 KB / 16 MB |
| Runtime RAM | <200 KB / 512 KB SRAM + 8 MB PSRAM |

## Raspberry Pi 3B deployment

### Inference benchmarks

| Metric | Value |
|---|---|
| TFLite Transformer standalone | ~60 ms |
| IF (50 trees) standalone | ~50 ms |
| Full server stack (FastAPI + simulator + WebSocket + both models) | ~1,400 ms |
| Sustainable sampling rate (full stack) | ~0.7 Hz |
| Default sampling rate | 1 Hz |

The gap between 60 ms standalone and 1,400 ms end-to-end comes from FastAPI overhead, WebSocket JSON serialization, NumPy array construction, and swap I/O when the 1 GB RAM is exhausted.

**Do not set the simulator above 1 Hz on Pi 3B.** The default was originally 10 Hz, which caused unbounded queue growth and system stalls. Set `samples_per_second=1`.

### Threading

The TFLite interpreter is not thread-safe. Concurrent `run_in_executor` calls from FastAPI cause `RuntimeError: There is at least 1 reference to internal data`. The fix is a `threading.Lock()` wrapping all interpreter calls in `detector.py`. This is already implemented.

### Memory footprint

| Component | Estimated |
|---|---|
| Python 3.11 + NumPy + scikit-learn + FastAPI | ~250 MB |
| TFLite interpreter + model weights | ~28 MB |
| IF model (loaded via joblib) | ~80 MB |
| OS and system services | ~250 MB |
| Sliding window buffer | ~0.02 MB |
| **Total** | **~508 MB / 1,024 MB physical** |

Swap activates during model loading and garbage collection pauses, causing latency spikes (2–3 s) on those samples.

## Project structure

```
scada_detector/
├── main.py                          # Entry point
├── app/
│   ├── api/
│   │   └── routes.py                # FastAPI endpoints + WebSocket
│   ├── inference/
│   │   └── detector.py              # SCADADetector ensemble (IF + TF-AE)
│   └── simulator/
│       └── data_generator.py        # 51-feature SCADA simulator, 5 attack types
├── templates/
│   └── dashboard.html               # Chart.js dashboard
├── static/                          # CSS/JS
├── saved_models/                    # Model files (not in repo)
├── calibrate_simulator.py           # IF training + scaler fitting
├── train_lstm_ae_simulator.py       # Transformer-AE training
└── edge/
    ├── scada_edge_detector.ino      # Arduino sketch for Yolo UNO
    └── scada_if_model.h             # 50-tree IF as C header (155 KB)
```

## Reproducibility checklist

If you are trying to replicate this on your own hardware:

1. **Pin your Python version to 3.11.** Check `ai-edge-litert` wheel availability for your platform before choosing an OS image.
2. **Pin scikit-learn to the exact version** used during training. Add it to `requirements.txt` with `==`, not `>=`.
3. **Use wall power on the Pi.** Not negotiable for write-heavy operations.
4. **Retrain the scaler on your data.** The StandardScaler must be fit on your normal operating data, not on SWaT data, unless you are running SWaT evaluation.
5. **Recalibrate thresholds.** Every data source has a different normal reconstruction error distribution. The provided thresholds (IF: -0.02, Transformer: 0.58) are tuned for the built-in simulator.
6. **If changing seq_len**, update it in the TFLite conversion script (TensorSpec shape), in `detector.py` (window buffer size), and in `data_generator.py` (window construction). All three must match.
7. **For ESP32 deployment**, reduce IF trees until the C header fits in flash. 50 trees / 155 KB worked for the Yolo UNO. Check your board's partition scheme.

## Known limitations

- **Deployed model not benchmarked:** The reduced Transformer-AE (seq_len=50) running on the Pi was not evaluated on SWaT. Detection quality likely degrades for subtle attacks.
- **No industrial protocol integration:** Data enters via HTTP/WebSocket, not OPC-UA, Modbus, or DNP3.
- **Single dataset:** Evaluated on SWaT only. The model must be retrained for any other SCADA system.
- **IF negligible in ensemble:** Isolation Forest contributed no alerts across five tested attack types. It remains useful for point anomalies (sensor malfunction) but not for deliberate temporal attacks.

## References

- SWaT Dataset: Mathur & Tippenhauer, "SWaT: A Water Treatment Testbed for Research and Training on ICS Security," 2016
- Isolation Forest: Liu, Ting, & Zhou, "Isolation Forest," IEEE ICDM, 2008
- Transformer: Vaswani et al., "Attention Is All You Need," NeurIPS, 2017
- TensorFlow Lite: https://www.tensorflow.org/lite
- emlearn: https://github.com/emlearn/emlearn
- OhStem Yolo UNO: https://ohstem.vn/product/yolo-uno/

## License

This project was developed as a capstone thesis at HCMUT. The SWaT dataset is provided by iTrust, SUTD, Singapore, under their research access terms.

## Author

Le Tien Phat (2252600) — Computer Engineering, HCMUT-VNUHCM
Supervisor: Dr. Vo Tuan Binh
