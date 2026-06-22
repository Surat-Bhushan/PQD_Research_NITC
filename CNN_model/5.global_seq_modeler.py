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
# TRANSFORMER POSITIONAL EMBEDDING LAYER
# ==========================================
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.pos_emb = keras.layers.Embedding(input_dim=sequence_length, output_dim=d_model)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = self.pos_emb(positions)
        return inputs + positions

# ==========================================
# STANDARD TIME-SERIES TRANSFORMER ENCODER
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Multi-Head Attention Block
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs  # First residual connection

    # Feed Forward Block
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res  # Second residual connection

# Input shape configuration
input_shape = (numOfFeatures, 1)  # (100, 1)
input_layer = keras.layers.Input(shape=input_shape)

# 1. Project local context features (Using kernel_size=3 instead of 1)
embedding_layer = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="linear")(input_layer)

# 2. Inject temporal sequence order indices
embedded_sequences = PositionalEmbedding(sequence_length=numOfFeatures, d_model=64)(embedding_layer)

# 3. Process via Stacked Attention Stages
x = transformer_encoder(embedded_sequences, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)

# 4. Global sequence pooling over structural dimensions
pooling_layer = keras.layers.GlobalAveragePooling1D()(x)

# 5. Classification Heads
dense_1 = keras.layers.Dense(units=128, activation='relu')(pooling_layer)
bn_output = keras.layers.BatchNormalization()(dense_1)
output_layer = keras.layers.Dense(units=17, activation='softmax')(bn_output)

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
