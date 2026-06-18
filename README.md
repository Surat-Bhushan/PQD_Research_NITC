# 📑 Table of Contents

* [Univariate Linear Regression using Gradient Descent](#univariate-linear-regression-using-gradient-descent)
  * [Objective](#objective)
  * [Dataset](#dataset)
  * [Linear Regression Model](#linear-regression-model)
  * [Loss Function](#loss-function)
  * [Gradient Descent](#gradient-descent)
  * [Training Procedure](#training-procedure)
  * [Visualization](#visualization)
  * [Prediction](#prediction)
  * [Verification using Scikit-Learn](#verification-using-scikit-learn)
  * [Experimental Observations and Parameter Tuning](#experimental-observations-and-parameter-tuning)
  * [Final Model Comparison](#final-model-comparison)
* [Multivariate Linear Regression on California Housing Dataset](#multivariate-linear-regression-on-california-housing-dataset)
  * [Objective](#objective-1)
  * [Dataset Description](#dataset-description)
  * [Model Summary](#model-summary)
  * [Learned Parameters](#learned-parameters)
  * [Regression Results](#regression-results)
  * [Classification Results](#classification-results)
  * [Predictions](#predictions)
  * [Conclusion](#conclusion)
* [Logistic Regression](#logistic-regression)
  * [Task Overview](#task-overview)
  * [Concept](#concept)
  * [Results](#results)
  * [Visualizations](#visualizations)
  * [Predictions for New Data](#predictions-for-new-data)
  * [Observations](#observations)
  * [Conclusion](#conclusion-1)
* [Machine Learning Classification Algorithms](#machine-learning-classification-algorithms)
  * [Naive Bayes Classifier](#naive-bayes-classifier)
  * [Decision Tree Classifier](#decision-tree-classifier)
  * [Random Forest Classifier](#random-forest-classifier)
* [Synthetic Signal Generation for Power Quality Disturbance Analysis](#synthetic-signal-generation-for-power-quality-disturbance-analysis)
* [Deep Learning Platform for Automatic Power Quality Disturbance (PQD) Classification](#deep-learning-platform-for-automatic-power-quality-disturbance-pqd-classification)
  * [Core References & Datasets](#-core-references--datasets)
  * [Dataset Specifications](#-dataset-specifications)
  * [Chronological Development & Architectural Evolution](#-chronological-development--architectural-evolution)
  * [Key Architectural Nuances](#%EF%B8%8F-key-architectural-nuances)
  * [Evolutionary Feature Comparison Matrix](#-evolutionary-feature-comparison-matrix)
  * [Evaluation Metrics (Pure CNN Baseline)](#-evaluation-metrics-pure-cnn-baseline)

---

# Univariate Linear Regression using Gradient Descent

## Objective

The objective of this task is to implement **Univariate Linear Regression** from scratch using **Gradient Descent**, analyze the effect of different learning rates on convergence, and compare the obtained results with Scikit-Learn's `LinearRegression` model.

---

## Dataset

The following data points are used:

```python
X = [0, 0.18, 0.26, 0.57, 0.48,
     0.62, 0.44, 0.55, 0.89,
     1.0, 0.92]
```

The target values are generated using:

```python
Y = 0.5 * X + 1 + 0.001 * randn()
```

where a small amount of random noise is added to simulate real-world observations.

---

## Linear Regression Model

The univariate linear regression model is:

y = mx + b

where:

- **m** = slope (weight)
- **b** = intercept (bias)

---

## Loss Function

Mean Squared Error (MSE) is used as the loss function:

<img width="165" height="38" alt="Screenshot 2026-05-24 at 10 35 03 PM" src="https://github.com/user-attachments/assets/4e3a6de4-c96d-47f7-873e-c1072cb6911e" />
The objective of training is to find values of m and b that minimize this loss.

---

## Gradient Descent
<img width="298" height="221" alt="Screenshot 2026-05-24 at 10 35 48 PM" src="https://github.com/user-attachments/assets/d861f89c-4d40-43d4-8372-1710def6530c" />

---

## Training Procedure

1. Initialize:
   - Weight (m) = 0
   - Bias (b) = 0

2. Train using Gradient Descent for multiple learning rates:

```python
[0.01, 0.05, 0.1, 0.5]
```

3. Run Gradient Descent for:
   - Maximum 1000 iterations
   - Or until convergence

4. Record:
   - Loss at every iteration
   - Final weight and bias
   - Number of iterations required for convergence

---

## Visualization

### 1. Loss vs Iterations

Shows the convergence behavior of different learning rates.

Observations:

- Smaller learning rates converge slowly.
- Larger learning rates converge faster.
- Excessively large learning rates may overshoot the optimum.
- An optimal learning rate achieves fast and stable convergence.

### 2. Regression Line

A scatter plot of the dataset along with the learned regression line is plotted to visualize the fit.

---

## Prediction

Using the model parameters obtained from the best learning rate, predictions are made for:

```python
x = 0.30
x = 0.75
```

---

## Verification using Scikit-Learn

The implementation is verified using:

```python
from sklearn.linear_model import LinearRegression
```

The following are compared:

- Weight (m)
- Bias (b)
- Predicted values

The closeness of these values validates the correctness of the Gradient Descent implementation.

---

## Experimental Observations and Parameter Tuning

During implementation, multiple experiments were performed to understand the effect of learning rate, convergence tolerance, and iteration limits.

---

### Experiment 1

**Maximum Iterations:** 1000  
**Tolerance:** 1e-8

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000014215 | 442 |
| 0.1 | 0.0000010616 | 246 |
| 0.5 | 0.0000007697 | 61 |

#### Observation

- Learning rate 0.5 achieved the lowest loss and fastest convergence.
- Learning rate 0.01 did not converge within 1000 iterations.
- This raised the question of whether 0.01 was truly worse or simply required more training.

---

### Experiment 2

**Maximum Iterations:** 5000  
**Tolerance:** 1e-8

#### Observation

The iteration limit was increased to investigate the behavior of the smaller learning rate.

Even with 5000 iterations, learning rate 0.01 was still being stopped prematurely by the convergence criterion before reaching a loss comparable to the other learning rates.

This suggested that the tolerance value itself might be causing early termination.

---

### Experiment 3

**Maximum Iterations:** 5000  
**Tolerance:** 1e-12

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000007109 | 4948 |
| 0.05 | 0.0000007106 | 1104 |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

#### Observation

- All learning rates converged to nearly the same minimum loss.
- Learning rate 0.01 was not inferior; it simply required substantially more iterations.
- The stricter tolerance prevented premature stopping and allowed optimization to approach the true optimum more closely.
- Learning rate 0.5 still reached the optimum much faster than the other learning rates.

---

### Experiment 4 (Final Configuration)

**Maximum Iterations:** 1000  
**Tolerance:** 1e-12

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000007108 | 1000 (did not converge) |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

#### Observation

- This configuration follows the assignment requirement of running for **1000 iterations or until convergence**.
- Learning rate 0.01 still required more than 1000 iterations and therefore did not fully converge.
- Learning rate 0.05 reached a loss very close to the optimum but also required slightly more than 1000 iterations for full convergence.
- Learning rates 0.1 and 0.5 successfully converged within the allowed iteration budget.
- Learning rate 0.5 achieved the same optimum while requiring the fewest iterations.

---

## Final Model Comparison

#### Gradient Descent (Best Learning Rate = 0.5)

```text
Weight = 0.498645
Bias   = 1.001412
```

#### Scikit-Learn Linear Regression

```text
Weight = 0.498636
Bias   = 1.001417
```

The difference is only in the fifth and sixth decimal places, confirming the correctness of the implementation.

---

### Predictions

#### Gradient Descent

| x | Predicted y |
|---|------------|
| 0.30 | 1.151005 |
| 0.75 | 1.375395 |

#### Scikit-Learn

| x | Predicted y |
|---|------------|
| 0.30 | 1.151008 |
| 0.75 | 1.375394 |

The predictions are nearly identical.

---

## Final Conclusion

The final implementation uses:

- Maximum Iterations = 1000
- Tolerance = 1e-12

This configuration was chosen because:

1. It satisfies the assignment requirement of running for **1000 iterations or until convergence**.
2. It avoids the premature stopping observed with a tolerance of 1e-8.
3. It allows converging learning rates to approach the optimum more accurately.
4. It produces results that closely match Scikit-Learn's `LinearRegression` model.

Among the tested learning rates, **α = 0.5** was selected as the best learning rate because it achieved essentially the same optimum as the smaller learning rates while converging in the fewest iterations.

## Plots Generated (Final Configuration)
<img width="743" height="471" alt="Screenshot 2026-05-24 at 10 52 40 PM" src="https://github.com/user-attachments/assets/0ddf318b-a7aa-42d6-8fc7-bfe2b81f5377" />
<img width="739" height="473" alt="Screenshot 2026-05-24 at 10 52 35 PM" src="https://github.com/user-attachments/assets/f758101c-c80e-472f-bf60-2301929ef609" />


# Multivariate Linear Regression on California Housing Dataset

## Objective

The objective of this task is to build a Multivariate Linear Regression model using the California Housing dataset to predict the median house value based on housing and demographic features. The model is trained using Scikit-Learn's `LinearRegression` class and evaluated using both regression metrics and classification metrics.

---

## Dataset Description

The dataset contains information about California housing block groups collected during the 1990 Census.

## Model Summary

### Dataset

- Total Samples: 17,000
- Training Samples: 13,600 (80%)
- Testing Samples: 3,400 (20%)

### Features Used

- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income

Excluded Features:

- longitude
- latitude

### Model

Multivariate Linear Regression:

```text
y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6 + b
```

where:

- y = predicted median house value
- wi = learned coefficient
- b = intercept

---

## Learned Parameters

| Feature | Coefficient |
|----------|-----------:|
| housing_median_age | 1849.79 |
| total_rooms | -20.65 |
| total_bedrooms | 95.97 |
| population | -32.53 |
| households | 128.47 |
| median_income | 47705.63 |

**Intercept:** -46,224.84

---

## Regression Results

| Metric | Value |
|----------|----------:|
| MAE | 55,065.78 |
| MSE | 5,558,804,420.35 |
| RMSE | 74,557.39 |
| R² Score | 0.5966 |

### Formulas

```text
MAE  = (1/n) Σ|yi - ŷi|

MSE  = (1/n) Σ(yi - ŷi)²

RMSE = √MSE

R² = 1 - (SSresidual / SStotal)

SSresidual = Σ(yi - ŷi)²

SStotal = Σ(yi - ȳ)²
```

### Observations

- Average prediction error ≈ $55,066 (MAE)
- Typical prediction error ≈ $74,557 (RMSE)
- Model explains 59.66% of variation in house values (R²)
- Median Income is the most influential feature

---

## Classification Results

To satisfy the assignment requirements, house prices were converted into two classes:

```text
Price > Median(y_test)  → High Price (1)

Price ≤ Median(y_test)  → Low Price (0)
```

The median was used because it creates relatively balanced classes and is less affected by outliers.

### Results

| Metric | Value |
|----------|----------:|
| Accuracy | 0.7912 |
| Precision | 0.7559 |
| Recall | 0.8600 |
| F1 Score | 0.8046 |

### Formulas

```text
Accuracy  = (TP + TN)/(TP + TN + FP + FN)

Precision = TP/(TP + FP)

Recall    = TP/(TP + FN)

F1 Score  = 2 × Precision × Recall /
            (Precision + Recall)
```

### Confusion Matrix

```text
[[1228  472]
 [ 238 1462]]
```

### Observations

- Classification Accuracy ≈ 79%
- Recall (86%) > Precision (75.6%)
- Model identifies high-value areas reasonably well
- More false positives (472) than false negatives (238)

---

## Predictions

| Data Point | Predicted House Value |
|------------|---------------------:|
| 1 | $426,695.76 |
| 2 | $224,966.17 |

### Observation

Data Point 1 received a significantly higher prediction primarily due to its higher median income.

---

## Conclusion

The model achieved moderate predictive performance with an R² score of 0.5966. Median Income emerged as the strongest predictor of house value, and the model produced reasonable predictions on unseen data.

## Scatter plot comparing actual vs. predicted house prices for the test set

<img width="770" height="564" alt="Screenshot 2026-05-25 at 1 06 16 AM" src="https://github.com/user-attachments/assets/26409ee7-9d2b-49e7-8956-73ecbda38b7b" />

# Logistic Regression

## Task Overview

The objective of this task was to build and compare Logistic Regression models for predicting passenger survival on the Titanic. Three models were developed:

- Full Feature Model
- 2-Feature Model (Age, Fare)
- 3-Feature Model (Age, Fare, Pclass)

The dataset was preprocessed by handling missing values, encoding categorical variables, removing irrelevant columns, and splitting the data into training and testing sets.

---

## Concept

Logistic Regression is a supervised machine learning algorithm used for binary classification problems. It predicts the probability of an observation belonging to a class using the Sigmoid Function.

- Probability ≥ 0.5 → Survived (1)
- Probability < 0.5 → Did Not Survive (0)

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---------|---------|---------|---------|---------|
| Full Feature Model | 81.01% | 78.57% | 74.32% | 76.39% |
| 2-Feature Model | 64.80% | 76.19% | 21.62% | 33.68% |
| 3-Feature Model | 73.74% | 76.47% | 52.70% | 62.40% |

---

## Visualizations

The following visualizations were generated:

- Confusion Matrix Heatmaps for all three models
- <img width="494" height="388" alt="Screenshot 2026-06-01 at 1 41 16 AM" src="https://github.com/user-attachments/assets/d737e629-fae2-43e3-8632-4a8f992cb58c" />
- <img width="477" height="389" alt="Screenshot 2026-06-01 at 1 41 31 AM" src="https://github.com/user-attachments/assets/301930cb-34c5-4fe5-a682-57e112e4f4a4" />
- <img width="487" height="395" alt="Screenshot 2026-06-01 at 1 41 24 AM" src="https://github.com/user-attachments/assets/bf6adea8-07d3-47af-bf40-387b285cfc84" />

- 2D Decision Boundary using Age and Fare
- <img width="742" height="573" alt="Screenshot 2026-06-01 at 1 41 41 AM" src="https://github.com/user-attachments/assets/7f205b04-7fb1-481e-a434-0158d62bcec6" />

- 3D Decision Plane using Age, Fare, and Pclass
<img width="742" height="573" alt="Screenshot 2026-06-01 at 1 41 41 AM" src="https://github.com/user-attachments/assets/514ac017-f67f-4aff-9b03-2d83706b5c63" />

---

## Predictions for New Data

### Passenger 1

| Model | Survival Probability | Prediction |
|---------|---------|---------|
| Full Feature | 0.0984 | Did Not Survive |
| 2-Feature | 0.3206 | Did Not Survive |
| 3-Feature | 0.2659 | Did Not Survive |

### Passenger 2

| Model | Survival Probability | Prediction |
|---------|---------|---------|
| Full Feature | 0.9134 | Survived |
| 2-Feature | 0.4843 | Did Not Survive |
| 3-Feature | 0.6192 | Survived |

---

## Observations

- The Full Feature Model achieved the highest accuracy and overall performance.
- The 3-Feature Model provided a balance between performance and visualization.
- The 2-Feature Model had the lowest accuracy because it used limited information.
- Passenger 2 was correctly classified by the Full Feature and 3-Feature models but misclassified by the 2-Feature model.
- Feature selection significantly affects prediction performance and model confidence.

---

## Conclusion

The Full Feature Logistic Regression model performed best because it utilized all relevant passenger information. Reduced-feature models made visualization easier but resulted in lower classification performance, demonstrating the importance of feature selection in machine learning.


# Machine Learning Classification Algorithms

This repository contains implementations of three fundamental machine learning classification algorithms using the Play Tennis dataset:

- Naive Bayes Classifier
- Decision Tree Classifier
- Random Forest Classifier

The goal is to understand how different classification algorithms learn patterns from data and make predictions.

---

## Naive Bayes Classifier

## What is Naive Bayes?

Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It predicts the most likely class of a sample by calculating probabilities from previously observed data.

For example, given weather conditions:

- Outlook = Sunny
- Temperature = Cool
- Humidity = High
- Wind = Strong

the algorithm estimates the probability of each class and predicts the class with the highest probability.

## Key Concepts

### Prior Probability

The probability of a class before observing any feature values.

Example:

P(Yes) = 9/14

P(No) = 5/14

### Conditional Probability

The probability of observing a feature value given a particular class.

Example:

P(Sunny | Yes)

### Posterior Probability

The probability of a class after observing all feature values.

Example:

P(Yes | X)

where X represents the complete set of features.

## Why is it Called Naive?

The algorithm assumes that all features are conditionally independent given the class label. Although this assumption is often unrealistic, the algorithm performs surprisingly well in practice.

## Advantages

- Simple and fast
- Easy to implement
- Works well with categorical data
- Effective on small datasets

---

## Decision Tree Classifier

## What is a Decision Tree?

A Decision Tree is a rule-based machine learning algorithm that makes predictions by repeatedly asking questions about the input features.

Example:

Outlook?

├── Overcast → Yes

├── Sunny

│   └── Humidity?

│       ├── High → No

│       └── Normal → Yes

└── Rain

    └── Wind?

        ├── Strong → No

        └── Weak → Yes

Each question splits the dataset into smaller and more homogeneous groups.

## Key Concepts

### Entropy

Entropy measures the uncertainty or impurity of a dataset.

- Entropy = 0 → Completely pure
- Higher entropy → More mixed classes

### Information Gain

Information Gain measures how much uncertainty is reduced after splitting on a feature.

The feature providing the highest information gain is selected for the split.

### Gini Impurity

Gini Impurity is another measure of impurity commonly used in decision trees.

Lower Gini values indicate better splits.

### Recursive Splitting

After selecting the best root node, the same process is repeated on each subset until the data becomes sufficiently pure.

## Advantages

- Easy to visualize and interpret
- Handles categorical data naturally
- Captures non-linear decision boundaries

---

## Random Forest Classifier

## What is Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make more reliable predictions.

Instead of relying on a single tree, several trees are trained independently and their predictions are combined using majority voting.

Example:

Tree 1 → Yes

Tree 2 → No

Tree 3 → Yes

Final Prediction → Yes

## Core Ideas Behind Random Forest

### Bootstrap Sampling

Each tree is trained on a randomly sampled version of the dataset using sampling with replacement.

This means:

- Some rows may appear multiple times
- Some rows may not appear at all

### Random Feature Selection

Different trees consider different subsets of features while creating splits.

This introduces diversity among the trees.

### Majority Voting

Each tree makes an individual prediction.

The class receiving the highest number of votes becomes the final prediction.

## Why is Random Forest Better Than a Single Tree?

A single decision tree can easily overfit the training data.

Random Forest reduces overfitting by averaging the predictions of many trees, resulting in:

- Lower variance
- Better generalization
- More stable predictions

## Advantages

- High predictive performance
- Less prone to overfitting
- Robust to noisy data
- Handles complex relationships effectively

---
# Synthetic Signal Generation for Power Quality Disturbance Analysis
Python_code.py contains the code for synthetic waveform generation. Below are the plots-
<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 13 53 PM" src="https://github.com/user-attachments/assets/ae8010dd-01d0-4f16-bea8-02c18fb8569d" />

<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 14 00 PM" src="https://github.com/user-attachments/assets/1d0999e1-57b1-4a6b-82d9-5acc08ccb9d5" />

<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 14 07 PM" src="https://github.com/user-attachments/assets/5c91ea84-f3f6-4448-81dc-822eb4f89271" />

# Deep Learning Platform for Automatic Power Quality Disturbance (PQD) Classification

This repository features a production-ready, scale-invariant 1D Convolutional Neural Network (CNN) optimized to extract complex time-frequency features from raw voltage waveforms and automatically classify them across 17 distinct operational categories.

---

## 📚 Core References & Datasets

* **Dataset Source:** [SEED Power Quality Disturbance Dataset (Kaggle)](https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset)
* **Architecture Basis:** Model 1 from the [chachkes247/Power-Quality-Disturbances GitHub Repository](https://github.com/chachkes247/Power-Quality-Disturbances)
* **Reference Paper:** *Deep Learning Algorithm for Automatic Classification of Power Quality Disturbances* by Fatema A. Albalooshi and M. R. Qader.

---

## 📈 Dataset Specifications

The model is evaluated using a perfectly balanced variations pipeline:
* **Total Scope:** 17,000 isolated signal waveforms (1,000 samples per class).
* **Signal Geometry:** 100 time-steps per sample, configured as a 3D tensor: `(Samples, 100, 1)`.
* **Target Classes (17):** Pure Sinusoidal, Sag, Swell, Interruption, Transient, Oscillatory Transient, Harmonics, Harmonics with Sag, Harmonics with Swell, Flicker, Flicker with Sag, Flicker with Swell, Sag with Oscillatory Transient, Swell with Oscillatory Transient, Sag with Harmonics, Swell with Harmonics, and Notch.

---

## ⏳ Chronological Development & Architectural Evolution

The codebase progressed through three major iterations to maximize feature resolution, validation integrity, and generalization safety.

### Step 1: The Initial Baseline (Lightweight Alternating CNN)
* **Design:** Alternating layout structure (`Conv1D` ➔ `MaxPooling1D` ➔ `Conv1D`).
* **Deficiencies:** Aggressive max-pooling strides cut down time-steps too quickly, erasing fine localized micro-anomalies (like sharp transient spikes or narrow notches). The terminal `Flatten()` layer tied the model to rigid input dimensions, causing compilation failures if the signal sequence window shifted.

### Step 2: Transition to Paper Architecture (Albalooshi & Qader)
* **Design:** Upgraded to a deep, VGG-style modular network using **Model 1** from the reference repository.
* **Why Model 1 over Model 2?** Model 1 provides a clean, pure spatial feature extraction baseline. Debugging and establishing peak performance on spatial filters first is critical before adding recurrent structures.
* **Advantage:** Replaced alternating blocks with **Stacked Convolutions** (`Conv1D` ➔ `Conv1D` ➔ `MaxPool1D`). Stacking consecutive convolutions expands the network's receptive field, allowing it to capture complex, non-linear waveshape interactions before data is lost to pooling.
* ** Future Roadmap Note:** A **BiLSTM (Bidirectional Long Short-Term Memory)** layer will be integrated directly beneath the final CNN unit block in the next phase. The CNN will process local geometric shapes while the BiLSTM tracks sequence context forwards and backwards, sharpening accuracy on complex hybrid categories (e.g., *Flicker_with_Sag*).

### Step 3: Overhauling the Repository Code (Final Production Script)
While the paper's architecture was powerful, the original repo implementation contained engineering flaws (naive index-slicing data splits, lack of training termination controls, and hardcoded `.mat` inputs). The original code was overhauled to create the final production-ready script through the following changes:

1.  **SEED Dataset Compatibility & CSV Ingestion:** Rewrote the data layer to dynamically parse 17 individual unheaded `.csv` signal sheets via `pd.read_csv(file_name, header=None)`. Expanded array stacking and channel configurations to scale from the original 16 target classes to 17.
2.  **Upgrading to a 3-Way Stratified Split:** The original repository used a basic, non-shuffled index slice to split data into Training and Validation sets, using the validation set for both training feedback and final score reports. The updated pipeline implements a rigorous double-stacked `train_test_split`:
    * **Train (70% - 11,900 samples):** Calculates loss gradients and updates weights.
    * **Validation (15% - 2,550 samples):** Evaluates mid-training optimization and guides callbacks.
    * **Test (15% - 2,550 samples):** Kept completely isolated in a separate vault. Evaluated exactly once after training ends to generate true performance metrics.
    * *Stratification ensures a uniform class distribution (700 / 150 / 150 samples per class) across all three splits.*
3.  **Coordinated Training Optimization Hooks:** The original repo relied on a fixed 43-epoch limit. The final script pairs two distinct training mechanics:
    * **Adaptive Learning Rate (The Steering Wheel):** Custom step-decay schedule starting at `alpha = 0.01` and cutting the rate in half every 10 epochs for tight convergence. This was present in the original repository code.
    * **Early Stopping (The Brakes):** Monitors validation loss with a roof of 100 epochs, `patience=10`, and `restore_best_weights=True`. If validation improvements stall, it cuts the loop and automatically rolls back model weights to the exact epoch that achieved peak performance.
4.  **Advanced Metrics Reports:** Integrated complete tracking systems post-training, outputting a precise text-based `classification_report` and a visual `seaborn` heatmapped confusion matrix on the unseen test dataset.

---

## 🛠️ Key Architectural Nuances

* **Scale Invariance via Global Max Pooling:** The legacy `Flatten()` layer was replaced with `GlobalMaxPooling1D()`. Instead of blindly smashing sequence indices together, global pooling scans each feature channel and extracts only its single highest activation value. This decouples the network from fixed input dimensions, allowing it to process 100-sample sequences or wider 640-sample sequences without dimensional errors. (Present in the original repo code)
* **Stride Invariance in Pooling:** Intermediate max-pooling layers use a custom configuration (`pool_size=3, strides=1`). This allows the filters to smooth local wave noise while keeping the spatial timeline resolution completely intact until the global pooling layer at the very end.(Present in the original repo code)

---

## 📊 Evolutionary Feature Comparison Matrix

| Design Vector | 1. Original Baseline | 2. Reference Repository Code | 3. Final Production Script |
| :--- | :--- | :--- | :--- |
| **Architectural Style** | Traditional / Lightweight | Deep Modular VGG-like | Deep Modular VGG-like |
| **Convolution Pattern**| Alternating (`Conv`➔`Pool`) | Stacked (`Conv`➔`Conv`➔`Pool`) | Stacked (`Conv`➔`Conv`➔`Pool`) |
| **Temporal Resolution**| Aggressive downsampling | Preserved resolution (`strides=1`) | Preserved resolution (`strides=1`) |
| **Input Flexibiliy** | Rigid (Hardcoded sequence size)| Scale-Invariant (`GlobalMaxPool`) | Scale-Invariant (`GlobalMaxPool`) |
| **Data Split Type** | 2-Way Random Split | Naive Fixed Index Slicing | **3-Way Stratified Split** |
| **Final Evaluation** | Shared Validation Split | Shared Validation Split | **Isolated Vault Test Set** |
| **Optimization Hook** | Static Learning Rate (Adam) | Custom Step-Decay Scheduler (Nadam)| **Step-Decay + Early Stopping + Weight Rollback** |
| **Target Scale** | Customized 17 Classes | Original 16 MATLAB Classes | **Customized 17 CSV Classes (SEED)** |


## 📊 Evaluation Metrics of Latest Run on Thu Jun 18 10:52:05 (Pure CNN Baseline)

Calculated entirely on the isolated, unseen **Test Set Vault** at peak performance weights:

* **True Model Test Accuracy:** 97.84%
* **Early Stopping Trigger:** Terminated at Epoch 60 (Restored to Epoch 50 weights)
* **Best Validation Loss:** 0.0769
* 

### Unbiased Test Classification Report
```text
                                  precision    recall  f1-score   support

                 Pure_Sinusoidal       1.00      1.00      1.00       150
                             Sag       0.97      1.00      0.99       150
                           Swell       0.97      1.00      0.99       150
                    Interruption       1.00      0.99      0.99       150
                       Transient       0.95      0.97      0.96       150
           Oscillatory_Transient       0.92      0.95      0.93       150
                       Harmonics       1.00      1.00      1.00       150
              Harmonics_with_Sag       0.99      0.98      0.98       150
            Harmonics_with_Swell       1.00      0.98      0.99       150
                         Flicker       0.99      1.00      1.00       150
                Flicker_with_Sag       0.95      0.98      0.97       150
              Flicker_with_Swell       0.99      0.97      0.98       150
  Sag_with_Oscillatory_Transient       0.98      0.92      0.95       150
Swell_with_Oscillatory_Transient       0.97      0.95      0.96       150
              Sag_with_Harmonics       0.98      0.99      0.98       150
            Swell_with_Harmonics       0.97      0.97      0.97       150
                           Notch       0.99      0.98      0.99       150

                        accuracy                           0.98      2550
                       macro avg       0.98      0.98      0.98      2550
                    weighted avg       0.98      0.98      0.98      2550
