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
The objective of training is to find values of **m** and **b** that minimize this loss.

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

