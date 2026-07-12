> **Note:** This repository contains the work completed during my **Summer Internship Programme 2026** at the **National Institute of Technology Calicut (NIT Calicut)** for the project **"Automatic Power Quality Disturbance Detection and Classification System."**
>
> The file **`Surat_Bhushan_Summer_Internship_Report_NITC.pdf`** contains the complete internship report, including the project background, methodology, implementation details, experimental results, conclusions, and learning outcomes.

# Table of Contents

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
* [Literature Review](#literature-review)
- [Comparative Analysis of the Five PQD Classification Models](#-comparative-analysis-of-the-five-pqd-classification-models)
  - [Model 1: 1D CNN](#model-1-1d-cnn)
    - [Overview](#overview)
    - [Evolution of the Architecture](#evolution-of-the-architecture)
    - [Model Structure](#model-structure)
    - [Performance](#performance)
  - [Model 2: Wavelet + Random Forest](#model-2-wavelet--random-forest)
    - [Overview](#overview-1)
    - [Model Structure](#model-structure-1)
    - [Key Characteristics](#key-characteristics)
    - [Performance](#performance-1)
  - [Model 3: Parallel CNN + BiLSTM](#model-3-parallel-cnn--bilstm)
    - [Overview](#overview-2)
    - [Model Structure](#model-structure-2)
    - [Key Characteristics](#key-characteristics-1)
    - [Performance](#performance-2)
  - [Model 4: CNN with Self-Attention](#model-4-cnn-with-self-attention)
    - [Overview](#overview-3)
    - [Model Structure](#model-structure-3)
    - [Key Characteristics](#key-characteristics-2)
    - [Performance](#performance-3)
  - [Model 5: Global Sequence Modeler](#model-5-global-sequence-modeler)
    - [Overview](#overview-4)
    - [Model Structure](#model-structure-4)
    - [Key Characteristics](#key-characteristics-3)
    - [Performance](#performance-4)
  - [Summary of the Five Models](#summary-of-the-five-models)
  - [Increasing Model Complexity](#increasing-model-complexity)
  - [Performance Summary](#performance-summary)
  - [Key Observations](#key-observations)
  - [Model Comparison Table](#model-comparison-table)
- [Structural Noise Robustness Analysis](#structural-noise-robustness-analysis)
- [Author](#author)

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
## Literature Review

To understand the research done in the domain of automatic power quality disturbance detection and classification, I read 18 research papers. Below is a summarised literature review of the same. For more detials kindly refer Complete_Document file present in this repository.

| Paper | Authors | Year | Methodology | Dataset | Key Contribution | Limitation |
|:------|:--------|:----:|:------------|:--------|:-----------------|:-----------|
| **Optimally Detecting and Classifying the Transmission Line Fault in Power System Using Hybrid Technique** | P. Rajesh, R. Kannan, J. Vishnupriyan, B. Rajani | 2022 | TSVD + HUA-Optimized RPNN | MATLAB/Simulink 400 kV transmission line dataset (11 fault classes) | Robust hybrid fault classification using TSVD feature extraction and optimized RPNN. | Evaluated only on simulated data; computationally expensive. |
| **A Critical Analysis of Methodologies for Detection and Classification of Power Quality Events in Smart Grid** | Rajender Kumar Beniwal, Manish Kumar Saini, Anand Nayyar, Basit Qureshi, Akanksha Aggarwal | 2021 | Comprehensive Survey | IEEE 1159 / IEC 61000-4-30 literature | Reviews conventional and deep learning PQD methods and identifies research gaps. | No proposed algorithm; survey paper only. |
| **A Generalized Classification Framework for Power Quality Disturbances Based on Synchrosqueezed Wavelet Transform and Convolutional Neural Networks** | Y. S. Upendra Vishwanath, S. Esakkirajan, B. Keerthiveena, Ram Bilas Pachori | 2023 | SWT + EfficientNetB0 CNN | IEEE 1159 synthetic dataset (15 classes) | Robust image-based PQD classification using Synchrosqueezed Wavelet Transform. | Performance decreases under extremely low SNR. |
| **A New Method With Hilbert Transform and Slip-SVD-Based Noise-Suppression Algorithm for Noisy Power Quality Monitoring** | Yan Wang, Qunzhan Li, Fulin Zhou, Yang Zhou, Xiuqing Mu | 2019 | Hilbert Transform + Slip-SVD | Synthetic & practical power-grid data | Rule-based PQ monitoring with effective noise suppression. | Depends on manually selected thresholds. |
| **A Novel Recognition Method for Complex Power Quality Disturbances Based on Visualization Trajectory Circle and Machine Vision** | Ding Yuan, Yulong Liu, Mingyang Lan, Tao Jin, Mohamed A. Mohamed | 2022 | Improved Hilbert Transform + Trajectory Circle + ResNet50 | Synthetic dataset + RT-LAB microgrid | Converts PQ signals into trajectory images for CNN classification. | Additional preprocessing overhead. |
| **A Systematic Review of Real-Time Detection and Classification of Power Quality Disturbances** | Systematic Review | 2023 | Literature Review | 179 papers selected from 4,068 studies | Reviews real-time PQD detection methods and future research directions. | No new algorithm proposed. |
| **Deep Learning Algorithm for Automatic Classification of Power Quality Disturbances** | Fatema A. Albalooshi, M. R. Qader | 2025 | Multi-scale 1D CNN + Attention | Synthetic MATLAB dataset | End-to-end feature learning without manual feature extraction. | Mostly validated on synthetic data. |
| **Complex PQD Classification Using Time-Frequency Analysis and Multiscale Parallel Attention Residual Network** | Jun Ma, Qiu Tang, Minjun He, Lorenzo Peretto, Zhaosheng Teng | 2024 | Modified Kaiser Window S-Transform (MKS) + MPARN | Synthetic 32-class dataset with hardware validation | Adaptive MKS and multiscale attention residual network for robust automatic classification of complex PQDs under noisy conditions. | Higher computational complexity and larger model size due to multiscale attention architecture. |
| **Detection and Classification of Multiple Power Quality Disturbances Using Stockwell Transform and Deep Learning** | Chenhui Cui, Yujie Duan, Hongli Hu, Liang Wang, Qing Liu | 2022 | Optimized Stockwell Transform + CNN | MATLAB-generated 37-class dataset | Accurate classification of multiple simultaneous disturbances. | Large image generation increases memory usage. |
| **Power Quality Disturbance Recognition Using Empirical Wavelet Transform and Feature Selection** | Shihan Chen, Zichie Li, Guobing Pan, Fang Xu | 2022 | EWT + ReliefF + SVM | IEEE 1159 synthetic dataset | Adaptive wavelet decomposition with feature selection. | Sensitive to decomposition parameters. |
| **Power Quality Disturbance Classification Based on Parallel Fusion of CNN and GRU** | Jiajun Cai, Kai Zhang, Hui Jiang | 2023 | Parallel CNN-GRU + Attention | IEEE 1159 synthetic dataset | Joint spatial and temporal feature extraction improves accuracy. | Higher computational cost than standalone models. |
| **A Comprehensive Review of Deep-Learning Applications to Power Quality Analysis** | Indu Sekhar Samanta, Subhasis Panda, Pravat Kumar Rout, Mohit Bajaj, et al. | 2023 | Review | IEEE 1159 benchmark datasets | Comprehensive overview of deep learning techniques for PQ analysis. | Mostly reviews synthetic-data-based research. |
| **Hierarchical Convolutional Neural Networks for Event Classification on PMU Measurements** | Martin Pavlovski, Mohammad Alqudah, Tatjana Dokic, Ameen Abdel Hai, Mladen Kezunovic, Zoran Obradovic | 2021 | Hierarchical CNN | Real-world PMU datasets | Efficient hierarchical event classification from PMU measurements. | Not designed for high-frequency PQ disturbances. |
| **Hybrid S-Transform and Kalman Filtering Approach for Detection and Measurement of Short Duration Disturbances in Power Networks** | P. K. Dash, M. V. Chilukuri | 2004 | S-Transform + Extended Kalman Filter | Simulated & laboratory validation | Early hybrid signal-processing approach for PQ monitoring. | Computationally intensive. |
| **Power Quality Disturbances Detection and Classification Based on Deep Convolution Auto-Encoder Networks** | Paras Khetarpal, Neelu Nagpal, Mohammed S. Al-Numay, Pierluigi Siano, Yogendra Arya, Neelam Kassarwani | 2023 | Deep Convolutional Sparse Autoencoder | Synthetic 15-class dataset | Learns compact latent representations automatically. | High memory requirement due to image conversion. |
| **Power Quality Disturbances Recognition via Lightweight Deep Learning Framework Integrated With Time-Frequency Analysis Embedded** | Jun Yang, Zhenyu Liu, Yingjie Lei, et al. | 2024 | MEMNet + Morlet Wavelet Convolution | Synthetic 32-class dataset | Lightweight embedded framework for real-time PQD recognition. | Sensitive to wavelet parameter initialization. |
| **Perception of Power Quality Disturbances Using Fourier, Short-Time Fourier, Continuous and Discrete Wavelet Transforms** | M. S. Priyadarshini, Mohit Bajaj, Lukas Prokop, Milijas Berahnu | 2024 | Comparative Transform Analysis | MATLAB-generated disturbance signals | Compares FT, STFT, CWT and DWT for PQ analysis. | No automated classification framework. |
| **Real-Time Detection and Classification of Power Quality Disturbances** | Mahsa Mozaffari, Keval Doshi, Yasin Yilmaz | 2022 | Sequential Multivariate CUSUM | Real-time multivariate grid monitoring data | Low-latency real-time PQD detection and classification. | Threshold tuning required to reduce false alarms. |

# Synthetic Signal Generation for Power Quality Disturbance Analysis
Python_code.py contains the code for synthetic waveform generation. Below are the plots-
<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 13 53 PM" src="https://github.com/user-attachments/assets/ae8010dd-01d0-4f16-bea8-02c18fb8569d" />

<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 14 00 PM" src="https://github.com/user-attachments/assets/1d0999e1-57b1-4a6b-82d9-5acc08ccb9d5" />

<img width="1280" height="800" alt="Screenshot 2026-06-10 at 5 14 07 PM" src="https://github.com/user-attachments/assets/5c91ea84-f3f6-4448-81dc-822eb4f89271" />

# 📊 Comparative Analysis of the Five PQD Classification Models

## Dataset

All five models were trained and evaluated on the **SEED Power Quality Disturbance (PQD) Dataset**, consisting of **17,000 balanced voltage waveforms** (1,000 samples for each of the 17 disturbance classes). A **70% / 15% / 15% stratified split** was used for training, validation, and testing to ensure fair and unbiased evaluation.

**Dataset:** https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset

---

# Model 1: 1D CNN

## Overview

A deep VGG-style 1D Convolutional Neural Network that performs end-to-end classification directly from raw voltage waveforms without manual feature engineering.

## Evolution of the Architecture

The final model was developed through three iterations:

- **Initial CNN:** A lightweight alternating `Conv1D → MaxPooling → Conv1D` design that suffered from excessive downsampling and rigid input dimensions.
- **Paper-based CNN:** Adopted the architecture from Albalooshi & Qader using stacked convolutional blocks, preserved temporal resolution, and a step-decay learning rate schedule that halves the learning rate every 10 epochs.
- **Final implementation:** Modified to support the 17-class SEED CSV dataset, introduced a **70/15/15 stratified train-validation-test split**, incorporated **Early Stopping with best-weight restoration**, generated detailed evaluation reports, and utilized **Global Max Pooling** for improved scale invariance and robustness.
- [* **Architecture Basis:** Model 1 from the [chachkes247/Power-Quality-Disturbances GitHub Repository](https://github.com/chachkes247/Power-Quality-Disturbances) * **Reference Paper:** *Deep Learning Algorithm for Automatic Classification of Power Quality Disturbances* by Fatema A. Albalooshi and M. R. Qader.]

## Model Structure

- Stacked Conv1D blocks with Batch Normalization
- ReLU activations
- Global Max Pooling
- Dense classification head
- Softmax output layer
- Nadam optimizer with dynamic learning rate scheduling

## Performance

- **Test Accuracy:** **98.16%**
- Excellent overall performance and one of the best-performing architectures in this study.

---

# Model 2: Wavelet + Random Forest

## Overview

A traditional machine learning pipeline that first extracts Discrete Wavelet Transform (DWT) features and then classifies them using a Random Forest ensemble.

## Model Structure

- 3-level DWT feature extraction (`db4`)
- Flattened wavelet coefficients
- Random Forest classifier with multiple decision trees

## Key Characteristics

- Explicit frequency-domain feature engineering
- Ensemble-based classification
- Fast inference and interpretable compared to deep networks

## Performance

- **Test Accuracy:** **91.22%**
- Performed well on many classes but was less effective than deep learning models for complex hybrid disturbances.

---

# Model 3: Parallel CNN + BiLSTM

## Overview

A hybrid deep learning architecture combining convolutional feature extraction with bidirectional sequence modeling.

## Model Structure

- Parallel CNN branch
- Parallel Bidirectional LSTM branch
- Feature fusion
- Dense classification layers

## Key Characteristics

- CNN extracts local waveform features
- BiLSTM captures long-range temporal dependencies in both directions
- Combines spatial and sequential information effectively

## Performance

- **Test Accuracy:** **98.27%**
- Highest-performing model among all five evaluated architectures.

---

# Model 4: CNN with Self-Attention

## Overview

An extension of the CNN architecture that incorporates a self-attention mechanism to emphasize the most informative regions of each waveform.

## Model Structure

- CNN feature extractor
- Self-Attention layer
- Dense classification head

## Key Characteristics

- Learns to focus on important temporal regions
- Improves feature weighting automatically
- Designed for complex PQD recognition

## Performance

- **Test Accuracy:** **91.18%**
- Achieved competitive performance but did not surpass the baseline CNN.

---

# Model 5: Global Sequence Modeler

## Overview

A Transformer-inspired architecture using positional embeddings and multi-head attention to model entire sequences globally.

## Model Structure

- Positional embeddings
- Multi-head self-attention blocks
- Residual connections
- Global pooling and dense classifier

## Key Characteristics

- Captures global dependencies across the signal
- Eliminates recurrent computations
- Attention-driven sequence modeling

## Performance

- **Test Accuracy:** **83.61%**
- Underperformed compared to CNN-based approaches on the relatively short 100-sample PQD sequences.

---

## Summary of the Five Models

| Model | Best At | Main Idea |
|--------|----------|-----------|
| **Model 1 – 1D CNN** | Learning local waveform patterns | Uses convolutional filters to automatically extract features from raw PQD signals. |
| **Model 2 – DWT + Random Forest** | Time-frequency analysis | Converts signals into wavelet features and classifies them using an ensemble of 200 decision trees. |
| **Model 3 – Parallel CNN + BiLSTM** | Combining spatial and temporal information | CNN learns local features while BiLSTM learns temporal dependencies simultaneously. |
| **Model 4 – CNN + Self-Attention** | Highlighting important signal regions | Uses attention to assign higher importance to informative parts of the waveform before classification. |
| **Model 5 – Transformer** | Global sequence modeling | Uses multi-head self-attention to learn relationships between all time steps in the signal simultaneously. |

---

### Increasing Model Complexity

```text
DWT + Random Forest
        │
        ▼
1D CNN
        │
        ▼
CNN + Self-Attention
        │
        ▼
CNN + BiLSTM
        │
        ▼
Transformer
```

- **Model 1** learns local spatial features.
- **Model 2** relies on handcrafted wavelet features instead of learned features.
- **Model 3** adds temporal sequence learning using BiLSTMs.
- **Model 4** adds attention to focus on important waveform regions.
- **Model 5** replaces convolution/LSTM-based sequence learning with global multi-head self-attention, allowing every time step to interact with every other time step.

---

# Performance Summary

| Model | Test Accuracy |
|--------|--------------:|
| Parallel CNN + BiLSTM | **98.27%** |
| 1D CNN | **98.16%** |
| Wavelet + Random Forest | **91.22%** |
| CNN + Self-Attention | **91.18%** |
| Global Sequence Modeler | **83.61%** |

## Key Observations

- **Parallel CNN + BiLSTM** achieved the highest classification accuracy.
- The **1D CNN** closely matched the best result while maintaining a simpler architecture.
- **Wavelet + Random Forest** demonstrated that handcrafted features remain competitive but lag behind deep learning on complex PQDs.
- **CNN with Self-Attention** performed similarly to the Random Forest pipeline but did not improve upon the baseline CNN.
- The **Global Sequence Modeler** was the weakest performer, indicating that attention-only architectures may require longer sequences or larger datasets for optimal effectiveness.

## Comparative Analysis of the Five Power Quality Disturbance (PQD) Classification Models

| Feature | **Model 1**<br>1D CNN | **Model 2**<br>DWT + Random Forest | **Model 3**<br>Parallel CNN + BiLSTM | **Model 4**<br>CNN + Self-Attention | **Model 5**<br>Transformer |
|---------|-------------------------|------------------------------------|--------------------------------------|--------------------------------------|-----------------------------|
| **Model Type** | Deep Learning | Traditional Machine Learning | Hybrid Deep Learning | Deep Learning | Deep Learning |
| **Feature Extraction** | Automatic (CNN) | Handcrafted (Wavelet Features) | CNN + BiLSTM | CNN + Self-Attention | Transformer Self-Attention |
| **Input** | Raw Signal | Raw Signal | Raw Signal | Raw Signal | Raw Signal |
| **Signal Length** | 100 Samples | 100 Samples | 100 Samples | 100 Samples | 100 Samples |
| **Number of Classes** | 17 | 17 | 17 | 17 | 17 |
| **Training Samples** | 17,000 | 17,000 | 17,000 | 17,000 | 17,000 |
| **Train / Val / Test Split** | 70% / 15% / 15% | 70% / 15% / 15% | 70% / 15% / 15% | 70% / 15% / 15% | 70% / 15% / 15% |
| **Main Learning Strategy** | Local feature extraction | Time-frequency feature extraction + Ensemble learning | Local + Temporal feature learning | Local feature extraction + Attention | Global sequence modeling |
| **Main Architecture** | Conv1D Network | Wavelet + Random Forest | Parallel CNN + BiLSTM | CNN + Self-Attention | Transformer Encoder |
| **CNN Filters** | 32 → 64 → 128 | — | 32 → 64 | 32 → 64 → 128 | Conv1D Embedding (64 Filters) |
| **Kernel Size** | 3 | — | 3 | 3 | 3 |
| **BiLSTM Units** | — | — | 32 → 64 | — | — |
| **Attention Mechanism** | No | No | No | Self-Attention | Multi-Head Self-Attention |
| **Transformer Encoder** | No | No | No | No | 2 Encoder Blocks |
| **Attention Heads** | — | — | — | — | 4 |
| **Head Size** | — | — | — | — | 64 |
| **Feed Forward Dimension** | — | — | — | — | 64 |
| **Wavelet Type** | — | Daubechies 4 (db4) | — | — | — |
| **Wavelet Decomposition Level** | — | 3 | — | — | — |
| **Random Forest Trees** | — | 200 | — | — | — |
| **Pooling Layer** | Global Max Pooling | — | Global Max Pooling | Global Average Pooling | Global Average Pooling |
| **Dense Layers** | 256 → 128 | — | 256 → 128 | 256 → 128 | 128 |
| **Output Layer** | Softmax (17) | Random Forest Voting | Softmax (17) | Softmax (17) | Softmax (17) |
| **Loss Function** | Categorical Cross-Entropy | — | Categorical Cross-Entropy | Categorical Cross-Entropy | Categorical Cross-Entropy |
| **Optimizer** | Nadam | — | Nadam | Nadam | Nadam |
| **Batch Size** | 64 | — | 64 | 64 | 64 |
| **Maximum Epochs** | 100 | — | 100 | 100 | 100 |
| **Learning Rate** | 0.01 (halved every 10 epochs) | — | 0.01 (halved every 10 epochs) | 0.01 (halved every 10 epochs) | 0.01 (halved every 10 epochs) |
| **Early Stopping** | Patience = 10 | Not Used | Patience = 10 | Patience = 10 | Patience = 10 |
| **Strength** | Learns local waveform patterns efficiently | Excellent time-frequency representation with fast training | Combines local and temporal information | Focuses on the most informative signal regions | Captures long-range dependencies across the entire signal |
| **Limitation** | Cannot model long-range dependencies well | Feature engineering required; cannot learn features automatically | Higher computational complexity | Attention increases computational cost | Highest computational complexity and training time |

## Terminal Graphs of Model3 -
<img width="473" height="311" alt="Screenshot 2026-07-07 at 4 21 35 PM" src="https://github.com/user-attachments/assets/e45a9a79-4910-4a28-8df2-c2d0af2adca3" />
<img width="877" height="564" alt="Screenshot 2026-07-07 at 4 21 47 PM" src="https://github.com/user-attachments/assets/79ba3d67-bf1c-4b6d-8bd9-dafed6cb1c8c" />

#### (Similarly there are outputs of rest of the 4 models.)
### The terminal output containing the detailed snapshot of model training and testing period is available in the file Termainl_Saved_Output.txt in this repo, for all five models. For in depth detial of model behaviour, that file can be referred

---
# Structural Noise Robustness Analysis

This section provides a performance evaluation and specifications summary of the structurally hardened **Parallel CNN || BiLSTM** model, trained on **unaugmented, raw seed data**.

To bridge the performance cliff observed in standard networks, the model's architecture was updated with wider receptive fields (`kernel_size=7`), `SpatialDropout1D`, and localized `GaussianNoise` injection layers to inherently smooth out stochastic variance.

best_model_with_noise_analysis.py file containts the parallel CNN BiLSTM model along with code for noise analysis.

---

## 📊 Noise Testing Specifications

| Parameter | Value / Range | Description |
| :--- | :--- | :--- |
| **Training Constraint** | Strict Raw Seed Data | No data augmentation or file-level preprocessing allowed |
| **Target Dataset** | Unseen Test Split (15%) | 2,550 isolated signal samples |
| **Signal Length ($N$)**| 100 Time Steps | 1D time-series vectors |
| **Noise Profile** | Gaussian Distribution ($\mu = 0$) | Zero-mean white noise scaled dynamically per sample |
| **Structural Fixes** | Macro-Kernels & Spatial Dropout | Replaced narrow filters to capture macro-contours |

---

## 🧮 Mathematical Framework

The AWGN generator dynamically calculates the instantaneous power of each individual waveform. This ensures that voltage drops (e.g., *Sags*, *Interruptions*) and high-energy transients are degraded proportionally without destroying low-amplitude signal definitions.

The noise power for any given target Signal-to-Noise Ratio (SNR in dB) is determined by:

1. **Signal Power ($P_{\text{signal}}$):**
   $$P_{\text{signal}} = \frac{1}{N} \sum_{t=1}^{N} x(t)^2$$

2. **Linear SNR Scale ($SNR_{\text{linear}}$):**
   $$SNR_{\text{linear}} = 10^{\left(\frac{SNR_{\text{dB}}}{10}\right)}$$

3. **Required Noise Power ($P_{\text{noise}}$):**
   $$P_{\text{noise}} = \frac{P_{\text{signal}}}{SNR_{\text{linear}}}$$

---

## 📝 Empirical Performance Log

* **Baseline Clean Test Accuracy:** `93.65%` (Weights restored from optimal Epoch 81)

| Evaluation Run | Noise Level (SNR) | Observed Test Accuracy | Operational Status |
| :---: | :---: | :---: | :--- |
| Control | **Clean (No Noise)** | **93.65%** | Nominal Benchmark |
| Run 5 | **50 dB** | **93.69%** | Fully Immune |
| Run 4 | **40 dB** | **93.73%** | Fully Immune |
| Run 3 | **30 dB** | **93.69%** | Fully Immune (Substation Standard) |
| Run 2 | **20 dB** | **81.14%** | Resilient / Active Industrial Deployment |
| Run 1 | **10 dB** | **23.53%** | Degraded (Severe Environmental Static) |

---

## 🔍 Validation & Architectural Insights

By altering the feature extraction mechanism instead of changing the source training data, the network achieved industrial-grade stability:

1. **Receptive Field Optimization:** Transitioning to a `kernel_size=7` configuration allowed the Conv1D paths to compute moving averages over raw data points. High-frequency noise is filtered out natively, letting the network maintain **>93% accuracy** down to a harsh $30\text{ dB}$ ambient environment.
2. **Mitigation of Sequential Fragility:** Incorporating Spatial Dropout and input layer Gaussian regularization prevented the BiLSTM path from memorizing hyper-precise synthetic trajectories. The network learned generalized geometric shapes rather than exact mathematical point paths.
3. **Convergence Dynamics:** Due to the heavier regularization framework, the network required more epochs to converge ($81$ epochs vs $32$ epochs originally). However, the resulting decision boundaries are significantly more robust, eliminating the fragile performance drop-offs seen in standard architectures.
<img width="772" height="485" alt="Screenshot 2026-07-07 at 6 26 56 AM" src="https://github.com/user-attachments/assets/7173ea86-a5c1-4c8e-98c8-fac2801f7802" />

# Author
## Surat Bhushan (2026)
(During internship at NIT Calicut, under the guidance of Dr. Gangireddy Narendra kumar Reddy, Assistant Professor in the Department of ECE, NIT Calicut.)
