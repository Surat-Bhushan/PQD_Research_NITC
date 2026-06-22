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
        # header=None prevents pandas from skipping the first signal row
        df = pd.read_csv(file_name, header=None)
        signals = df.values  # Shape: (1000, 100)
        
        X_list.append(signals)
        # Generate matching numerical target labels (0 to 16)
        Y_list.append(np.full((signals.shape[0], 1), class_idx))
        print(f" Successfully loaded {file_name} | Shape: {signals.shape}")
    else:
        print(f"⚠️ Warning: {file_name} not found in the current folder!")

# Stack individual class arrays into master matrices
X_all = np.vstack(X_list)  # Combined shape: (17000, 100)
Y_all = np.vstack(Y_list)  # Combined shape: (17000, 1)

# Add the explicit 3rd dimension channel for Conv1D -> (Samples, Time Steps, Channels)
X_all = np.expand_dims(X_all, axis=-1)
numOfFeatures = X_all.shape[1]  # Evaluates to 100

print(f"\nDataset fully prepared: X shape = {X_all.shape} | Y shape = {Y_all.shape}")

# ==========================================
# 3. ONE-HOT ENCODING & 3-WAY STRATIFIED SPLIT
# ==========================================
OHE = OneHotEncoder(sparse_output=False)
Y_all_OneHot = OHE.fit_transform(Y_all)

# Split 1: Isolate the absolute final Test Set (15%) from the temporary train/val pool
X_train_val, X_test, Y_train_val_OneHot, Y_test_OneHot = train_test_split(
    X_all, Y_all_OneHot, test_size=0.15, random_state=42, stratify=Y_all
)

# Extract integer labels from the temporary pool to maintain stratification
Y_train_val_ints = np.argmax(Y_train_val_OneHot, axis=1)

# Split 2: Divide remaining 85% into Train (70% total) and Validation (15% total)
X_train, X_validate, Y_train_OneHot, Y_validate_OneHot = train_test_split(
    X_train_val, Y_train_val_OneHot, test_size=0.1765, random_state=42, stratify=Y_train_val_ints
)

# Store flat integer labels for evaluation steps
Y_test_true = np.argmax(Y_test_OneHot, axis=1)

print(f"\n--- Cross-Validation Split Complete ---")
print(f"Training matrices shape (70%):   {X_train.shape}")
print(f"Validation matrices shape (15%): {X_validate.shape}")
print(f"Test matrices shape (15%):       {X_test.shape}\n")

# ==========================================
# REPLACE SECTION 4 FOR MODEL 3: PARALLEL CNN || BiLSTM
# ==========================================
input_shape = (numOfFeatures, 1)
input_layer = keras.layers.Input(shape=input_shape)

# PATH A: Spatial Convolutional Feature Extraction Block
conv_1 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu')(input_layer)
conv_2 = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu')(conv_1)
pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=1)(conv_2)
bn_1   = keras.layers.BatchNormalization()(pool_1)

conv_3 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')(bn_1)
conv_4 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')(conv_3)
pool_2 = keras.layers.MaxPool1D(pool_size=3, strides=1)(conv_4)
bn_2   = keras.layers.BatchNormalization()(pool_2)

spatial_flat = keras.layers.GlobalMaxPooling1D()(bn_2)

# PATH B: Temporal Sequential Feature Extraction Block
# Two stacked BiLSTMs processing macro contours directly from the raw data
lstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True))(input_layer)
lstm_2 = keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=False))(lstm_1)
temporal_flat = keras.layers.BatchNormalization()(lstm_2)

# FUSION: Merge Spatial and Temporal features side-by-side
merged_features = keras.layers.concatenate([spatial_flat, temporal_flat])

# Dense Classification Sub-Network
dense_1 = keras.layers.Dense(units=256, activation='relu')(merged_features)
dense_2 = keras.layers.Dense(units=128, activation='relu')(dense_1)
bn_output = keras.layers.BatchNormalization()(dense_2)

output_layer = keras.layers.Dense(units=17, activation='softmax')(bn_output)

# Instantiate the Dual-Path Functional Model
model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.summary()

# ==========================================
# 5. CALLBACKS CONFIGURATION & TRAINING
# ==========================================
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stop_callback = EarlyStopping(
    monitor='val_loss',         # Tracks the validation loss
    patience=10,                # Number of epochs to wait before killing execution
    restore_best_weights=True,  # Discards overfitting and restores your optimal epoch
    verbose=1
)

print("\nStarting model training...")
model_history = model.fit(
    X_train, Y_train_OneHot, 
    batch_size=64,
    epochs=100,  # Cap set to 100; early stopping will manage the optimal end point
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
plt.title('CNN: Accuracy vs. Number of Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 7. VAULT EVALUATION (UNSEEN TEST DATA)
# ==========================================
score = model.evaluate(X_test, Y_test_OneHot, verbose=0)
print("\n==============================================")
print(f"TRUE Model Test Accuracy: {score[1]*100:.2f}%")
print("==============================================")

# Single isolated sample inference check from the Test Partition
Xtest_sample = np.array(X_test[62:63])
Ytest_sample = Y_test_true[62:63]

XtestPredict = model.predict(Xtest_sample, verbose=0)
Xtest_label_idx = np.argmax(XtestPredict, axis=1)[0]
actual_label_idx = Ytest_sample[0]

print("\n--- Live Test Sample Verification ---")
print(f"Predicted Class: Index {Xtest_label_idx} -> ({classes[Xtest_label_idx]})")
print(f"Actual Ground Truth: Index {actual_label_idx} -> ({classes[actual_label_idx]})")
print("Is prediction correct?", Xtest_label_idx == actual_label_idx)

plt.figure(figsize=(6, 3))
plt.plot(Xtest_sample[0])
plt.title(f"Visualized Test Waveform Sample: {classes[actual_label_idx]}")
plt.grid(True)
plt.show()

# ==========================================
# 8. PERFORMANCE REPORTS & CONFUSION MATRIX
# ==========================================
print("\nGenerating Unbiased Performance Metrics on all TEST signals...")

# Generate predictions on the completely unseen Test Set
Y_pred_one_hot = model.predict(X_test, verbose=0)
Y_pred_classes = np.argmax(Y_pred_one_hot, axis=1)

# Complete Classification Summary
report = classification_report(Y_test_true, Y_pred_classes, target_names=classes)
print("\n================== FINAL TEST CLASSIFICATION REPORT ==================")
print(report)
print("======================================================================")

# Generate and Plot the Heatmap Confusion Matrix
cm = confusion_matrix(Y_test_true, Y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True,          
    fmt='d',             
    cmap='Blues',        
    xticklabels=classes, 
    yticklabels=classes  
)
plt.title('Power Quality Disturbances - Final Test Confusion Matrix')
plt.ylabel('Actual True Class (Unseen Test Data)')
plt.xlabel('Model Predicted Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
