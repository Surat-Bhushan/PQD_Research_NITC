import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CUSTOM LEARNING RATE SCHEDULER
# ==========================================
def scheduler(epoch):
    dropEvery = 10
    initAlpha = 0.01
    factor = 0.5
    exp = np.floor((1 + epoch) / dropEvery)
    alpha = initAlpha * (factor ** exp)
    print(f'lr = {alpha}')
    return float(alpha)

# ==========================================
# 2. DYNAMICALLY LOAD THE 17 CSV FILES
# ==========================================
classes = [
    "Pure_Sinusoidal", "Sag", "Swell", "Interruption", "Transient",
    "Oscillatory_Transient", "Harmonics", "Harmonics_with_Sag", "Harmonics_with_Swell",
    "Flicker", "Flicker_with_Sag", "Flicker_with_Swell", "Sag_with_Oscillatory_Transient",
    "Swell_with_Oscillatory_Transient", "Sag_with_Harmonics", "Swell_with_Harmonics", "Notch"
]

X_list = []
Y_list = []

print("Reading CSV data files...")
for class_idx, class_name in enumerate(classes):
    file_name = f"{class_name}.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, header=None)
        signals = df.values  # Shape: (1000, 100)
        
        X_list.append(signals)
        Y_list.append(np.full((signals.shape[0], 1), class_idx))
        print(f" Successfully loaded {file_name} | Shape: {signals.shape}")
    else:
        print(f"⚠️ Warning: {file_name} not found in the current folder!")

X_all = np.vstack(X_list)  # Combined shape: (17000, 100)
Y_all = np.vstack(Y_list)  # Combined shape: (17000, 1)

X_all = np.expand_dims(X_all, axis=-1)
numOfFeatures = X_all.shape[1]  # Evaluates to 100

print(f"\nDataset fully prepared: X shape = {X_all.shape} | Y shape = {Y_all.shape}")

# ==========================================
# 3. ONE-HOT ENCODING & 3-WAY STRATIFIED SPLIT
# ==========================================
OHE = OneHotEncoder(sparse_output=False)
Y_all_OneHot = OHE.fit_transform(Y_all)

X_train_val, X_test, Y_train_val_OneHot, Y_test_OneHot = train_test_split(
    X_all, Y_all_OneHot, test_size=0.15, random_state=42, stratify=Y_all
)

Y_train_val_ints = np.argmax(Y_train_val_OneHot, axis=1)

X_train, X_validate, Y_train_OneHot, Y_validate_OneHot = train_test_split(
    X_train_val, Y_train_val_OneHot, test_size=0.1765, random_state=42, stratify=Y_train_val_ints
)

Y_test_true = np.argmax(Y_test_OneHot, axis=1)

print(f"\n--- Cross-Validation Split Complete ---")
print(f"Training matrices shape (70%):   {X_train.shape}")
print(f"Validation matrices shape (15%): {X_validate.shape}")
print(f"Test matrices shape (15%):       {X_test.shape}\n")

# ==========================================
# 4. STRUCTURALLY ROBUST MODEL ARCHITECTURE
# ==========================================
input_shape = (numOfFeatures, 1)
input_layer = keras.layers.Input(shape=input_shape)

# Internal regularizer to protect features against jittering during training
noise_input = keras.layers.GaussianNoise(0.05)(input_layer)

# PATH A: Spatial Convolutional Feature Extraction Block (Widened Kernels)
conv_1 = keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(noise_input)
conv_2 = keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(conv_1)
pool_1 = keras.layers.MaxPool1D(pool_size=2, strides=2)(conv_2)
bn_1   = keras.layers.BatchNormalization()(pool_1)
drop_s1 = keras.layers.SpatialDropout1D(0.2)(bn_1) # Forces feature map redundancy

conv_3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(drop_s1)
conv_4 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(conv_3)
pool_2 = keras.layers.MaxPool1D(pool_size=2, strides=2)(conv_4)
bn_2   = keras.layers.BatchNormalization()(pool_2)
drop_s2 = keras.layers.SpatialDropout1D(0.2)(bn_2)

spatial_flat = keras.layers.GlobalMaxPooling1D()(drop_s2)

# PATH B: Temporal Sequential Feature Extraction Block
lstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True))(noise_input)
lstm_2 = keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=False))(lstm_1)
temporal_flat = keras.layers.BatchNormalization()(lstm_2)
drop_t  = keras.layers.Dropout(0.3)(temporal_flat)

# FUSION: Merge Spatial and Temporal features side-by-side
merged_features = keras.layers.concatenate([spatial_flat, drop_t])

# Dense Classification Sub-Network
dense_1 = keras.layers.Dense(units=256, activation='relu')(merged_features)
dense_1_drop = keras.layers.Dropout(0.3)(dense_1)
dense_2 = keras.layers.Dense(units=128, activation='relu')(dense_1_drop)
bn_output = keras.layers.BatchNormalization()(dense_2)

output_layer = keras.layers.Dense(units=17, activation='softmax')(bn_output)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.summary()

# ==========================================
# 5. CALLBACKS CONFIGURATION & TRAINING
# ==========================================
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stop_callback = EarlyStopping(
    monitor='val_loss',         
    patience=12,  # Slighly extended patience to accommodate structural updates              
    restore_best_weights=True,  
    verbose=1
)

print("\nStarting model training...")
model_history = model.fit(
    X_train, Y_train_OneHot, 
    batch_size=64,
    epochs=100,  
    callbacks=[lr_callback, early_stop_callback],
    validation_data=(X_validate, Y_validate_OneHot),
    verbose=1
)

# ==========================================
# 6. PLOT ACCURACY HISTORY
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Robust Parallel CNN-BiLSTM: Accuracy vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 7. VAULT EVALUATION (UNSEEN TEST DATA)
# ==========================================
score = model.evaluate(X_test, Y_test_OneHot, verbose=0)
print("\n==============================================")
print(f"TRUE Model Test Accuracy (Clean): {score[1]*100:.2f}%")
print("==============================================")

# ==========================================
# 8. ROBUSTNESS & NOISE ANALYSIS (AWGN)
# ==========================================
print("\nInitializing Noise Robustness Analysis...")

def add_awgn_to_dataset(X_data, snr_db):
    X_noisy = np.zeros_like(X_data)
    for i in range(X_data.shape[0]):
        signal = X_data[i, :, 0]
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear if signal_power > 0 else 1e-4
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        X_noisy[i, :, 0] = signal + noise
    return X_noisy

snr_levels = [10, 20, 30, 40, 50]
noise_accuracies = []

for snr in snr_levels:
    X_test_noisy = add_awgn_to_dataset(X_test, snr)
    metrics = model.evaluate(X_test_noisy, Y_test_OneHot, verbose=0)
    acc_percentage = metrics[1] * 100
    noise_accuracies.append(acc_percentage)
    print(f" -> Test Accuracy at {snr} dB SNR: {acc_percentage:.2f}%")

# --- PLOT THE SNR ROBUSTNESS CURVE ---
plt.figure(figsize=(8, 5))
plt.plot(snr_levels, noise_accuracies, marker='o', linestyle='-', color='crimson', linewidth=2)
plt.axhline(y=score[1]*100, color='darkgreen', linestyle='--', label=f"Clean Test Acc ({score[1]*100:.2f}%)")
plt.title('Model Degradation Curve under AWGN Stress (Robust Architecture)')
plt.xlabel('Signal-to-Noise Ratio (SNR) in dB (Higher = Cleaner)')
plt.ylabel('Test Accuracy (%)')
plt.xticks(snr_levels)
plt.ylim(0, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.show()