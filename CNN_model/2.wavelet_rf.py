import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

import pywt
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. DATASET CONFIGURATION
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

# Stack individual class arrays into master matrices
X_all = np.vstack(X_list)  # Combined shape: (17000, 100)
Y_all = np.vstack(Y_list).squeeze()  # Shape: (17000,)

print(f"\nDataset fully prepared: X shape = {X_all.shape} | Y shape = {Y_all.shape}")

# ==========================================
# 2. 3-WAY STRATIFIED SPLIT (INTEGER LABELS)
# ==========================================
# Split 1: Isolate the final Test Set (15%)
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X_all, Y_all, test_size=0.15, random_state=42, stratify=Y_all
)

# Split 2: Divide remaining 85% into Train (70% total) and Validation (15% total)
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X_train_val, Y_train_val, test_size=0.1765, random_state=42, stratify=Y_train_val
)

print(f"\n--- Cross-Validation Split Complete ---")
print(f"Training signals shape (70%):   {X_train.shape}")
print(f"Validation signals shape (15%): {X_validate.shape}")
print(f"Test signals shape (15%):       {X_test.shape}\n")

# ==========================================
# 3. DISCRETE WAVELET TRANSFORM (DWT) FEATURE EXTRACTION
# ==========================================
print("--- Extracting Discrete Wavelet Transform (DWT) Features ---")

def extract_wavelet_features(X_data):
    features_list = []
    for signal in X_data:
        # Perform a 3-level decomposition using Daubechies 4 ('db4') wavelet
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        # Flatten all approximation and detail coefficients into a single structural feature vector
        feature_vector = np.concatenate([c for c in coeffs])
        features_list.append(feature_vector)
    return np.array(features_list)

# Transform raw signals into time-frequency multi-scale tabular vectors
X_train_wavelet = extract_wavelet_features(X_train)
X_val_wavelet = extract_wavelet_features(X_validate)
X_test_wavelet = extract_wavelet_features(X_test)

print(f"Original Train Shape: {X_train.shape} -> Transformed Wavelet Feature Shape: {X_train_wavelet.shape}")

# ==========================================
# 4. RANDOM FOREST TRAINING & MODEL EVALUATION
# ==========================================
print("\nTraining Random Forest Classifier Ensemble...")
# Initialize Random Forest with 200 estimators utilizing all processor cores (-1)
ml_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
ml_model.fit(X_train_wavelet, Y_train)

# Calculate performance checks
train_acc = accuracy_score(Y_train, ml_model.predict(X_train_wavelet))
val_acc = accuracy_score(Y_validate, ml_model.predict(X_val_wavelet))
test_acc = accuracy_score(Y_test, ml_model.predict(X_test_wavelet))

print(f"Finished Fitting! Training Accuracy: {train_acc*100:.2f}% | Validation Accuracy: {val_acc*100:.2f}%")

# ==========================================
# 5. VAULT EVALUATION (UNSEEN TEST DATA)
# ==========================================
print("\n==============================================")
print(f"TRUE Model Test Accuracy: {test_acc*100:.2f}%")
print("==============================================")

# Single isolated sample inference check from the Test Partition
sample_idx = 62
Xtest_sample_raw = X_test[sample_idx]
Xtest_sample_wavelet = X_test_wavelet[sample_idx].reshape(1, -1)
actual_label_idx = Y_test[sample_idx]

# Predict the class index
predicted_label_idx = ml_model.predict(Xtest_sample_wavelet)[0]

print("\n--- Live Test Sample Verification ---")
print(f"Predicted Class: Index {predicted_label_idx} -> ({classes[predicted_label_idx]})")
print(f"Actual Ground Truth: Index {actual_label_idx} -> ({classes[actual_label_idx]})")
print("Is prediction correct?", predicted_label_idx == actual_label_idx)

plt.figure(figsize=(6, 3))
plt.plot(Xtest_sample_raw)
plt.title(f"Visualized Test Waveform Sample: {classes[actual_label_idx]}")
plt.grid(True)
plt.show()

# ==========================================
# 6. PERFORMANCE REPORTS & CONFUSION MATRIX
# ==========================================
print("\nGenerating Unbiased Performance Metrics on all TEST signals...")

# Generate predictions on the completely unseen Test Set
Y_pred_classes = ml_model.predict(X_test_wavelet)

# Complete Classification Summary
report = classification_report(Y_test, Y_pred_classes, target_names=classes)
print("\n================== FINAL TEST CLASSIFICATION REPORT ==================")
print(report)
print("======================================================================")

# Generate and Plot the Heatmap Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True,          
    fmt='d',             
    cmap='Blues',        
    xticklabels=classes, 
    yticklabels=classes  
)
plt.title('DWT + Random Forest - Final Test Confusion Matrix')
plt.ylabel('Actual True Class (Unseen Test Data)')
plt.xlabel('Model Predicted Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()