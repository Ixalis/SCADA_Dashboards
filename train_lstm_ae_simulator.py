#!/usr/bin/env python3
"""
LSTM-AE Training on Simulator Data
====================================
Same architecture as LSTM_AE_Phase2_FINAL.py but trained on
data from the SCADA simulator instead of SWaT.

Run on your Mac:
    cd /Users/ixa/Phase_2/scada_detector
    python train_lstm_ae_simulator.py

Output:
    saved_models/lstm_ae_sim.keras      - Trained model
    saved_models/scaler_sim.pkl         - Simulator scaler
    saved_models/sim_ae_threshold.txt   - Recommended threshold
"""

import numpy as np
import joblib
import os
import sys
import time

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed,
    Dropout, BatchNormalization, Bidirectional, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

# Import simulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.simulator.data_generator import SCADASimulator, AttackType

print("=" * 60)
print("LSTM-AE Training on Simulator Data")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# =============================================================================
# CONFIG
# =============================================================================
SEQUENCE_LENGTH = 50
N_FEATURES = 51
LATENT_DIM = 64
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
LSTM_UNITS_3 = 32
DROPOUT_RATE = 0.3
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 0.001
CLIP_NORMAL = 5.0
N_NORMAL_SAMPLES = 20000  # Generate 20k normal samples
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

np.random.seed(42)
tf.random.set_seed(42)
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# ATTENTION LAYER (same as Phase 2)
# =============================================================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return x * a

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# =============================================================================
# 1. GENERATE SIMULATOR DATA
# =============================================================================
print(f"\n[1/6] Generating {N_NORMAL_SAMPLES} normal simulator samples...")
sim = SCADASimulator(n_features=N_FEATURES, seed=42)
normal_data = []
for i in range(N_NORMAL_SAMPLES):
    sample, _, _ = sim.generate_sample()
    normal_data.append(sample)
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{N_NORMAL_SAMPLES}")
normal_data = np.array(normal_data, dtype=np.float32)
print(f"  Shape: {normal_data.shape}")

# =============================================================================
# 2. SCALE DATA
# =============================================================================
print(f"\n[2/6] Fitting StandardScaler...")
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_data).astype(np.float32)
normal_scaled = np.clip(normal_scaled, -CLIP_NORMAL, CLIP_NORMAL)
print(f"  Scaled range: [{normal_scaled.min():.2f}, {normal_scaled.max():.2f}]")

# Save scaler
scaler_path = os.path.join(SAVE_DIR, "scaler_sim.pkl")
joblib.dump(scaler, scaler_path)
print(f"  Saved scaler to: {scaler_path}")

# =============================================================================
# 3. CREATE SEQUENCES
# =============================================================================
print(f"\n[3/6] Creating sequences (length={SEQUENCE_LENGTH})...")
sequences = []
for i in range(len(normal_scaled) - SEQUENCE_LENGTH + 1):
    sequences.append(normal_scaled[i:i + SEQUENCE_LENGTH])
sequences = np.array(sequences, dtype=np.float32)
print(f"  Sequences shape: {sequences.shape}")

# Train/val split
split = int(0.85 * len(sequences))
X_train = sequences[:split]
X_val = sequences[split:]
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

# =============================================================================
# 4. BUILD MODEL (same architecture as Phase 2)
# =============================================================================
print(f"\n[4/6] Building LSTM-AE model...")

inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES), name='input')

# Encoder
x = Bidirectional(
    LSTM(LSTM_UNITS_1, activation='tanh', return_sequences=True),
    name='encoder_bilstm_1'
)(inputs)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)

x = Bidirectional(
    LSTM(LSTM_UNITS_2, activation='tanh', return_sequences=True),
    name='encoder_bilstm_2'
)(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)

x = AttentionLayer(name='attention')(x)

x = LSTM(LSTM_UNITS_3, activation='tanh', return_sequences=False,
         name='encoder_lstm_3')(x)
x = Dropout(DROPOUT_RATE)(x)

latent = Dense(LATENT_DIM, activation='tanh', name='latent_space')(x)

# Decoder
x = RepeatVector(SEQUENCE_LENGTH, name='repeat_vector')(latent)

x = LSTM(LSTM_UNITS_3, activation='tanh', return_sequences=True,
         name='decoder_lstm_1')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)

x = LSTM(LSTM_UNITS_2, activation='tanh', return_sequences=True,
         name='decoder_lstm_2')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)

x = LSTM(LSTM_UNITS_1, activation='tanh', return_sequences=True,
         name='decoder_lstm_3')(x)
x = Dropout(DROPOUT_RATE)(x)

outputs = TimeDistributed(
    Dense(N_FEATURES, activation='linear'),
    name='output'
)(x)

model = Model(inputs=inputs, outputs=outputs, name='LSTM_AE_Simulator')
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mae', metrics=['mse'])
model.summary()

total_params = model.count_params()
print(f"\n  Total parameters: {total_params:,}")

# =============================================================================
# 5. TRAIN
# =============================================================================
print(f"\n[5/6] Training for up to {EPOCHS} epochs...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, 'lstm_ae_sim_best.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

start_time = time.time()
history = model.fit(
    X_train, X_train,  # autoencoder: input = target
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_time
print(f"\n  Training completed in {training_time/60:.1f} minutes")
print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss:   {history.history['val_loss'][-1]:.6f}")

# =============================================================================
# 6. CALIBRATE THRESHOLD
# =============================================================================
print(f"\n[6/6] Calibrating detection threshold...")

# Compute reconstruction errors on normal validation data
print("  Computing reconstruction errors on validation data...")
val_reconstructed = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
val_errors = np.mean(np.abs(X_val - val_reconstructed), axis=(1, 2))

p95 = np.percentile(val_errors, 95)
p99 = np.percentile(val_errors, 99)
p995 = np.percentile(val_errors, 99.5)

print(f"  Normal reconstruction error stats:")
print(f"    Mean:  {val_errors.mean():.6f}")
print(f"    Std:   {val_errors.std():.6f}")
print(f"    P95:   {p95:.6f}")
print(f"    P99:   {p99:.6f}")
print(f"    P99.5: {p995:.6f}")

# Test on attack data
print("\n  Testing on attack data...")
attack_results = []
for atype, name, sensor, intensity in [
    (AttackType.POINT_SPIKE,       "Point Spike",      0, 10.0),
    (AttackType.SLOW_DRIFT,        "Slow Drift",       0, 10.0),
    (AttackType.NOISE_INJECTION,   "Noise Injection",  0, 10.0),
    (AttackType.FROZEN,            "Frozen Sensor",     0, 1.0),
    (AttackType.REPLAY,            "Replay",            0, 1.0),
    (AttackType.CORRELATION_BREAK, "Corr Break",        0, 1.0),
]:
    sim_t = SCADASimulator(n_features=N_FEATURES, seed=99)
    # Generate enough for sequences
    raw_samples = []
    # 200 normal warmup
    for _ in range(200):
        s, _, _ = sim_t.generate_sample()
        raw_samples.append(s)
    # 100 attack samples
    sim_t.start_attack(atype, duration_samples=100, target_sensor=sensor, intensity=intensity)
    for _ in range(100):
        s, _, _ = sim_t.generate_sample()
        raw_samples.append(s)

    raw_samples = np.array(raw_samples, dtype=np.float32)
    scaled = scaler.transform(raw_samples)
    scaled = np.clip(scaled, -CLIP_NORMAL, CLIP_NORMAL).astype(np.float32)

    # Create attack sequences (from the attack portion)
    attack_seqs = []
    for i in range(200, len(scaled) - SEQUENCE_LENGTH + 1):
        attack_seqs.append(scaled[i:i + SEQUENCE_LENGTH])
    if len(attack_seqs) == 0:
        # If not enough, use what we can
        attack_seqs.append(scaled[-SEQUENCE_LENGTH:])
    attack_seqs = np.array(attack_seqs, dtype=np.float32)

    recon = model.predict(attack_seqs, batch_size=BATCH_SIZE, verbose=0)
    errors = np.mean(np.abs(attack_seqs - recon), axis=(1, 2))

    det95 = (errors > p95).sum()
    det99 = (errors > p99).sum()
    total = len(errors)
    print(f"    {name:20s}: mean_err={errors.mean():.6f}, det@P95={det95}/{total} ({det95/total*100:.0f}%), det@P99={det99}/{total} ({det99/total*100:.0f}%)")

# Save model
model_path = os.path.join(SAVE_DIR, "lstm_ae_sim.keras")
model.save(model_path)
print(f"\n  Model saved to: {model_path}")

# Save threshold info
thresh_path = os.path.join(SAVE_DIR, "sim_ae_threshold.txt")
with open(thresh_path, 'w') as f:
    f.write(f"p95={p95:.6f}\n")
    f.write(f"p99={p99:.6f}\n")
    f.write(f"p995={p995:.6f}\n")
    f.write(f"mean={val_errors.mean():.6f}\n")
    f.write(f"std={val_errors.std():.6f}\n")
print(f"  Thresholds saved to: {thresh_path}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"""
  Model:     {model_path}
  Scaler:    {scaler_path}
  Threshold: P99 = {p99:.6f} (recommended)

  To use with dashboard:

    SCADA_IF_THRESHOLD=-0.02 SCADA_TRANSFORMER_THRESHOLD={p99:.6f} python main.py \\
      --if-model saved_models/isolation_forest_sim_calibrated.pkl \\
      --transformer-model saved_models/lstm_ae_sim.keras \\
      --scaler saved_models/scaler_sim.pkl

  NOTE: Both IF and Transformer now use the simulator scaler (scaler_sim.pkl).
  The --scaler flag should point to scaler_sim.pkl, NOT the SWaT scaler.

  BUT IMPORTANT: detector.py has a bug where load_transformer() overwrites
  the IF scaler. Since both models now use the SAME scaler (scaler_sim.pkl),
  this is no longer a problem!
""")
